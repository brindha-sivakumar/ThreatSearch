
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from nlp import process_line


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth(
    technique_cve_map_path: str,
    attack_path:            str,
    min_relevant:           int = 2,
) -> list[dict]:
    """
    Build evaluation queries from ATT&CK technique names + their mapped CVEs.

    Returns list of:
        { "query_id": "T1021.002",
          "query_text": "SMB/Windows Admin Shares",
          "relevant_ids": {"CVE-2017-0144", ...} }

    Only techniques with >= min_relevant CVE mappings are included
    (too few relevant docs makes P@K uninformative).
    """
    # Load technique names
    technique_names = {}
    if os.path.exists(attack_path):
        with open(attack_path, encoding="utf-8") as f:
            bundle = json.load(f)
        objects = bundle if isinstance(bundle, list) else bundle.get("objects", [])
        for obj in objects:
            if obj.get("type") != "attack-pattern":
                continue
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    tid  = ref.get("external_id", "").upper()
                    name = obj.get("name", "")
                    if tid and name:
                        technique_names[tid] = name

    # Load technique→CVE mapping
    if not os.path.exists(technique_cve_map_path):
        raise FileNotFoundError(
            f"Technique→CVE map not found at {technique_cve_map_path}.\n"
            "Run: python expander.py build-map"
        )
    with open(technique_cve_map_path, encoding="utf-8") as f:
        mapping = json.load(f)

    queries = []
    for tid, cve_ids in mapping.items():
        if len(cve_ids) < min_relevant:
            continue
        name = technique_names.get(tid.upper(), tid)
        queries.append({
            "query_id"    : tid.upper(),
            "query_text"  : name,
            "relevant_ids": set(c.upper() for c in cve_ids),
        })

    print(f"[eval] {len(queries)} evaluation queries loaded", file=sys.stderr)
    return queries


# ── Metric functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    top_k = [r.upper() for r in retrieved[:k]]
    hits  = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id.upper() in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id.upper() in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    actual_dcg  = dcg_at_k(retrieved, relevant, k)
    # Ideal DCG: all relevant docs at the top
    ideal_hits  = min(len(relevant), k)
    ideal_dcg   = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_results(retrieved: list[str], relevant: set, k: int) -> dict:
    return {
        f"P@{k}"    : precision_at_k(retrieved, relevant, k),
        "MRR"       : reciprocal_rank(retrieved, relevant),
        f"nDCG@{k}" : ndcg_at_k(retrieved, relevant, k),
    }


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(
    queries:          list[dict],
    scorer:           str,
    index_dir:        str,
    doc_lengths_path: str,
    lsa_dir:          str,
    dictionary_path:  str,
    attack_path:      str,
    nvd_map_path:     str,
    top_k:            int,
) -> dict:
    """
    Run all queries through the specified scorer, return aggregate metrics.
    """
    from expander import QueryExpander

    # Load scorer(s)
    if scorer in ("bm25", "tfidf"):
        from ranker import Ranker
        engine = Ranker(index_dir, doc_lengths_path, scorer=scorer)
        use_lsa = False
    else:
        from lsa_ranker import LSARanker
        engine  = LSARanker(lsa_dir, dictionary_path)
        use_lsa = True

    # Load expander (shared across scorers)
    expander = None
    if os.path.exists(attack_path):
        expander = QueryExpander(attack_path,
                                 nvd_map_path if os.path.exists(nvd_map_path) else None)

    all_metrics: list[dict] = []
    timings: list[float]    = []

    print(f"\n[eval] running {len(queries)} queries with scorer={scorer} …", file=sys.stderr)

    for q in queries:
        _, terms = process_line("Q " + q["query_text"])
        if not terms:
            continue

        # Expand query
        expanded_terms = terms
        term_weights   = None
        if expander:
            expanded_terms, term_weights = expander.expand(terms)

        # Retrieve
        t0 = time.perf_counter()
        if use_lsa:
            results = engine.query(expanded_terms, top_n=top_k, term_weights=term_weights)
        else:
            results = engine.score(expanded_terms, top_n=top_k, term_weights=term_weights)
        timings.append(time.perf_counter() - t0)

        retrieved = [r[0] for r in results]
        metrics   = evaluate_results(retrieved, q["relevant_ids"], top_k)
        metrics["query_id"]   = q["query_id"]
        metrics["query_text"] = q["query_text"]
        metrics["n_relevant"] = len(q["relevant_ids"])
        all_metrics.append(metrics)

    if not all_metrics:
        print("[eval] WARNING: no queries produced results.", file=sys.stderr)
        return {}

    # Aggregate
    p_key    = f"P@{top_k}"
    ndcg_key = f"nDCG@{top_k}"
    agg = {
        "scorer"         : scorer,
        "n_queries"      : len(all_metrics),
        f"MAP@{top_k}"   : float(np.mean([m[p_key]    for m in all_metrics])),
        "MRR"            : float(np.mean([m["MRR"]    for m in all_metrics])),
        f"nDCG@{top_k}"  : float(np.mean([m[ndcg_key] for m in all_metrics])),
        "avg_latency_ms" : float(np.mean(timings) * 1000),
        "per_query"      : all_metrics,
    }
    return agg


# ── Report writer ─────────────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict], top_k: int):
    p_key    = f"MAP@{top_k}"
    ndcg_key = f"nDCG@{top_k}"
    print("\n── Evaluation Results ──────────────────────────────────────────")
    print(f"{'Scorer':<10}  {p_key:<10}  {'MRR':<10}  {ndcg_key:<12}  {'Latency':>10}")
    print("─" * 62)
    for scorer, r in results.items():
        if not r:
            continue
        print(
            f"{scorer:<10}  "
            f"{r[p_key]:<10.4f}  "
            f"{r['MRR']:<10.4f}  "
            f"{r[ndcg_key]:<12.4f}  "
            f"{r['avg_latency_ms']:>8.1f} ms"
        )
    print()


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Full JSON (per-query breakdowns)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        # Convert sets to lists for JSON serialisation
        json.dump(results, f, indent=2, default=list)

    # Summary CSV for easy import into reports
    rows = ["scorer,MAP,MRR,nDCG,latency_ms,n_queries"]
    for scorer, r in results.items():
        if not r:
            continue
        k = next(k for k in r if k.startswith("MAP"))
        nd = next(k for k in r if k.startswith("nDCG"))
        rows.append(
            f"{scorer},"
            f"{r[k]:.4f},"
            f"{r['MRR']:.4f},"
            f"{r[nd]:.4f},"
            f"{r['avg_latency_ms']:.1f},"
            f"{r['n_queries']}"
        )
    csv_path = os.path.join(out_dir, "eval_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(rows))

    print(f"[eval] results saved → {out_dir}/eval_results.json  +  eval_summary.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ThreatSearch evaluation framework")
    ap.add_argument("--index-dir",    default="data/index")
    ap.add_argument("--doc-lengths",  default="data/index/doc_lengths.json")
    ap.add_argument("--lsa-dir",      default="data/lsa_index")
    ap.add_argument("--dict",         default="data/index/dictionary.txt")
    ap.add_argument("--attack",       default="data/attack/enterprise-attack.json")
    ap.add_argument("--nvd-map",      default="data/index/technique_cve_map.json")
    ap.add_argument("--top-k",        default=10, type=int)
    ap.add_argument("--min-relevant", default=2,  type=int)
    ap.add_argument("--scorer",       default="both",
                    choices=["bm25", "tfidf", "lsa", "both"],
                    help="Which scorer(s) to evaluate")
    ap.add_argument("--out-dir",      default="data/eval")
    args = ap.parse_args()

    # Load ground truth
    queries = load_ground_truth(
        args.nvd_map, args.attack, args.min_relevant
    )
    if not queries:
        print("[eval] No evaluation queries. Run: python expander.py build-map", file=sys.stderr)
        sys.exit(1)

    # Determine which scorers to run
    scorers_to_run = ["bm25", "lsa"] if args.scorer == "both" else [args.scorer]

    all_results = {}
    for scorer in scorers_to_run:
        # Skip LSA if lsa_index not built
        if scorer == "lsa" and not os.path.exists(
            os.path.join(args.lsa_dir, "doc_matrix.npy")
        ):
            print(f"[eval] skipping LSA — {args.lsa_dir} not found. Run lsa_build.py first.",
                  file=sys.stderr)
            continue

        all_results[scorer] = run_evaluation(
            queries          = queries,
            scorer           = scorer,
            index_dir        = args.index_dir,
            doc_lengths_path = args.doc_lengths,
            lsa_dir          = args.lsa_dir,
            dictionary_path  = args.dict,
            attack_path      = args.attack,
            nvd_map_path     = args.nvd_map,
            top_k            = args.top_k,
        )

    print_comparison(all_results, args.top_k)
    save_results(all_results, args.out_dir)
