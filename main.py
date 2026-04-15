import argparse
import json
import os
import re
import sys
import time

def p(data_dir: str, *parts: str) -> str:
    return os.path.join(data_dir, *parts)


def check_data(data_dir: str) -> bool:
    """Verify NVD feeds and ATT&CK bundle exist before starting."""
    ok = True
    nvd_dir = p(data_dir, "nvd")
    if not os.path.isdir(nvd_dir) or not any(
        f.endswith((".json", ".json.gz")) for f in os.listdir(nvd_dir)
    ):
        print(f"[error] NVD feeds not found in {nvd_dir}/", file=sys.stderr)
        ok = False

    attack = p(data_dir, "attack", "enterprise-attack.json")
    if not os.path.exists(attack):
        print(f"[error] ATT&CK bundle not found: {attack}", file=sys.stderr)
        ok = False

    return ok


def check_index(data_dir: str) -> bool:
    """Check that a built index exists — required for query mode."""
    index_dir = p(data_dir, "index")
    return (
        os.path.exists(p(data_dir, "index", "dictionary.txt")) and
        os.path.exists(p(data_dir, "index", "doc_lengths.json")) and
        os.path.isdir(index_dir) and
        any(f.startswith("index_") for f in os.listdir(index_dir))
    )


def check_lsa(data_dir: str) -> bool:
    return (
        os.path.exists(p(data_dir, "lsa_index", "term_matrix.npy")) and
        os.path.exists(p(data_dir, "lsa_index", "doc_matrix.npy"))
    )


# ── Phase runners ────────────────────────────

def run_ingest(data_dir: str) -> bool:
    print("\n[phase 1] ingestion — NVD + ATT&CK -> corpus shards")
    sys.path.insert(0, os.path.dirname(__file__) or ".")
    from ingest import ingest
    try:
        ingest(
            nvd_dir     = p(data_dir, "nvd"),
            attack_path = p(data_dir, "attack", "enterprise-attack.json"),
            out_dir     = p(data_dir, "corpus"),
        )
        return True
    except Exception as e:
        print(f"[error] ingest failed: {e}", file=sys.stderr)
        return False


def run_index(data_dir: str) -> bool:
    print("\n[phase 1] index — two-pass inverted index")
    from index import build_index
    try:
        build_index(
            corpus_dir = p(data_dir, "corpus"),
            out_dir    = p(data_dir, "index"),
        )
    except Exception as e:
        print(f"[error] index failed: {e}", file=sys.stderr)
        return False

    print("\n[phase 1] merge — k-way merge into single index file")
    from merge_index import merge
    try:
        merge(index_dir=p(data_dir, "index"))
    except Exception as e:
        print(f"[error] merge failed: {e}", file=sys.stderr)
        return False

    print("\n[phase 1] doc lengths — precompute for BM25")
    import run_doc_lengths
    try:
        lengths = run_doc_lengths.compute_doc_lengths(p(data_dir, "corpus"))
        out = p(data_dir, "index", "doc_lengths.json")
        with open(out, "w") as f:
            json.dump(lengths, f)
        print(f"[run_doc_lengths] {len(lengths):,} documents -> {out}")
    except Exception as e:
        print(f"[error] doc_lengths failed: {e}", file=sys.stderr)
        return False

    print("\n[phase 1] expander — build technique->CVE map")
    from expander import build_technique_cve_map
    try:
        build_technique_cve_map(
            attack_path = p(data_dir, "attack", "enterprise-attack.json"),
            corpus_dir  = p(data_dir, "corpus"),
            out_path    = p(data_dir, "index", "technique_cve_map.json"),
        )
    except Exception as e:
        print(f"[error] expander build-map failed: {e}", file=sys.stderr)
        return False

    return True


def run_lsa(data_dir: str, components: int = 300) -> bool:
    print(f"\n[lsa] building latent semantic index (k={components})")
    from lsa_build import build
    try:
        build(
            index_dir        = p(data_dir, "index"),
            dictionary_path  = p(data_dir, "index", "dictionary.txt"),
            doc_lengths_path = p(data_dir, "index", "doc_lengths.json"),
            out_dir          = p(data_dir, "lsa_index"),
            n_components     = components,
        )
        return True
    except Exception as e:
        print(f"[error] lsa_build failed: {e}", file=sys.stderr)
        return False


def run_lda(data_dir: str, topics: int = 20, passes: int = 10) -> bool:
    print(f"\n[phase 3] LDA topic modeling ({topics} topics, {passes} passes)")
    from topic_model import run as lda_run
    try:
        lda_run(
            corpus_dir  = p(data_dir, "corpus"),
            out_dir     = p(data_dir, "lda"),
            n_topics    = topics,
            passes      = passes,
            n_top_words = 15,
            min_doc_len = 5,
        )
        return True
    except Exception as e:
        print(f"[error] topic_model failed: {e}", file=sys.stderr)
        return False


def run_viz(data_dir: str) -> bool:
    print("\n[phase 3] visualization — word clouds + tactic heatmap")
    try:
        from visualize import (
            load_topic_terms, load_doc_topics,
            plot_wordclouds, plot_tactic_heatmap,
        )
        os.makedirs(p(data_dir, "viz"), exist_ok=True)
        topics     = load_topic_terms(p(data_dir, "lda"))
        doc_topics = load_doc_topics(p(data_dir, "lda"))
        plot_wordclouds(topics, p(data_dir, "viz", "topic_wordclouds.png"))
        plot_tactic_heatmap(
            p(data_dir, "lda"),
            p(data_dir, "attack", "enterprise-attack.json"),
            p(data_dir, "index", "technique_cve_map.json"),
            p(data_dir, "viz", "tactic_heatmap.png"),
        )
        return True
    except Exception as e:
        print(f"[warning] visualization failed: {e}", file=sys.stderr)
        return False


def run_eval(data_dir: str) -> bool:
    print("\n[phase 3] evaluation — BM25 vs LSA (P@K, MRR, nDCG@K)")
    from evaluate import run_evaluation, save_results, print_comparison, load_ground_truth
    import os as _os
 
    attack_path = p(data_dir, "attack", "enterprise-attack.json")
    nvd_map     = p(data_dir, "index", "technique_cve_map.json")
    index_dir   = p(data_dir, "index")
    lsa_dir     = p(data_dir, "lsa_index")
 
    try:
        queries = load_ground_truth(nvd_map, attack_path, min_relevant=2)
        if not queries:
            print("[eval] no evaluation queries — expander map may be empty", file=sys.stderr)
            return False
 
        all_results = {}
        for scorer in ["bm25", "lsa"]:
            if scorer == "lsa" and not check_lsa(data_dir):
                print("[eval] skipping LSA — lsa_index not built", file=sys.stderr)
                continue
            all_results[scorer] = run_evaluation(
                queries          = queries,
                scorer           = scorer,
                index_dir        = index_dir,
                doc_lengths_path = p(index_dir, "doc_lengths.json"),
                lsa_dir          = lsa_dir,
                dictionary_path  = p(index_dir, "dictionary.txt"),
                attack_path      = attack_path,
                nvd_map_path     = nvd_map,
                top_k            = 10,
            )
 
        print_comparison(all_results, top_k=10)
        save_results(all_results, out_dir=p(data_dir, "eval"))
        return True
    except Exception as e:
        print(f"[error] evaluate failed: {e}", file=sys.stderr)
        return False



def run_query(data_dir: str, query: str, scorer: str, top: int, no_expand: bool):
    """Run a single query against the built index and print results."""
    if not check_index(data_dir):
        print("[error] Index not found. Run the full pipeline first.", file=sys.stderr)
        sys.exit(1)

    if scorer == "lsa" and not check_lsa(data_dir):
        print("[error] LSA index not found. Run without --skip-lsa first.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, os.path.dirname(__file__) or ".")
    from nlp      import process_line
    from expander import QueryExpander
    from ranker   import Ranker

    index_dir = p(data_dir, "index")
    attack    = p(data_dir, "attack", "enterprise-attack.json")
    nvd_map   = p(data_dir, "index", "technique_cve_map.json")

    # Boolean pre-filter
    boolean_filter_ids = None
    if re.search(r"\b(AND|OR|NOT)\b", query, re.IGNORECASE):
        from boolean_query import BooleanQueryHandler
        boolean_filter_ids = BooleanQueryHandler(index_dir).execute(query)
        if not boolean_filter_ids:
            print("Boolean filter matched 0 documents.")
            return
        print(f"Boolean filter: {len(boolean_filter_ids):,} candidate docs")

    # NLP
    clean = re.sub(r"\b(AND|OR|NOT)\b", " ", query, flags=re.IGNORECASE)
    _, terms = process_line("Q " + clean)
    if not terms:
        print("No indexable terms after NLP processing.")
        return

    # Expansion
    expanded_terms = terms
    term_weights   = None
    if not no_expand and os.path.exists(attack):
        exp = QueryExpander(
            attack_path = attack,
            nvd_mapping = nvd_map if os.path.exists(nvd_map) else None,
        )
        expanded_terms, term_weights = exp.expand(terms)

    # Score
    t0 = time.perf_counter()
    if scorer == "lsa":
        from lsa_ranker import LSARanker
        results = LSARanker(
            p(data_dir, "lsa_index"),
            p(index_dir, "dictionary.txt"),
        ).query(expanded_terms, top_n=top, term_weights=term_weights)
        if boolean_filter_ids:
            results = [(d, s, src) for d, s, src in results if d in boolean_filter_ids]
    else:
        results = Ranker(
            index_dir, p(index_dir, "doc_lengths.json"), scorer
        ).score(
            expanded_terms, top_n=top,
            term_weights=term_weights,
            candidate_ids=boolean_filter_ids,
        )
    elapsed = time.perf_counter() - t0

    # Print results
    _CVE    = "https://nvd.nist.gov/vuln/detail/"
    _ATTACK = "https://attack.mitre.org/techniques/"

    def url(doc_id: str) -> str:
        if doc_id.upper().startswith("CVE-"):
            return _CVE + doc_id.upper()
        if doc_id.upper().startswith("T") and doc_id[1:].replace(".", "").isdigit():
            return _ATTACK + doc_id.upper().replace(".", "/") + "/"
        return ""

    print(f"\nQuery : {query}")
    print(f"Terms : {terms}")
    print(f"Scorer: {scorer}  |  {elapsed*1000:.1f} ms  |  {len(results)} result(s)\n")

    if results:
        col = max(len(r[0]) for r in results) + 2
        print(f"{'Rank':<5}  {'Score':>8}  {'Source':<7}  {'Doc ID':<{col}}  URL")
        print("─" * 72)
        for rank, (doc_id, score, source) in enumerate(results, 1):
            src = {"nvd": "NVD", "attack": "ATT&CK"}.get(source, source)
            print(f"{rank:<5}  {score:>8.4f}  {src:<7}  {doc_id:<{col}}  {url(doc_id)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ThreatSearch — pipeline runner and search interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir",       required=True,
                    help="Directory containing nvd/ and attack/ subdirectories")
    ap.add_argument("--query",          default=None,
                    help="Run this query and exit (skips pipeline build)")
    ap.add_argument("--skip-ingest",    action="store_true")
    ap.add_argument("--skip-index",     action="store_true")
    ap.add_argument("--skip-lsa",       action="store_true")
    ap.add_argument("--skip-lda",       action="store_true")
    ap.add_argument("--skip-eval",      action="store_true")
    ap.add_argument("--lsa-components", type=int, default=300)
    ap.add_argument("--lda-topics",     type=int, default=20)
    ap.add_argument("--lda-passes",     type=int, default=10)
    ap.add_argument("--scorer",         default="bm25",
                    choices=["bm25", "tfidf", "lsa"])
    ap.add_argument("--top",            type=int, default=10)
    ap.add_argument("--no-expand",      action="store_true")

    args     = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # Query-only shortcut
    if args.query:
        run_query(data_dir, args.query, args.scorer, args.top, args.no_expand)
        return

    # Full pipeline
    if not check_data(data_dir):
        sys.exit(1)

    t0       = time.perf_counter()
    failures = []

    if not args.skip_ingest and not run_ingest(data_dir):
        failures.append("ingest")

    if not args.skip_index and not run_index(data_dir):
        failures.append("index")

    if not args.skip_lsa and not run_lsa(data_dir, components=args.lsa_components):
        failures.append("lsa")

    if not args.skip_lda:
        if not run_lda(data_dir, topics=args.lda_topics, passes=args.lda_passes):
            failures.append("lda")
        else:
            run_viz(data_dir)

    if not args.skip_eval and not run_eval(data_dir):
        failures.append("eval")

    elapsed = time.perf_counter() - t0
    print(f"\n[done] {elapsed:.1f}s", end="")
    if failures:
        print(f"  — errors in: {', '.join(failures)}")
    else:
        print("  — all phases completed")


if __name__ == "__main__":
    main()