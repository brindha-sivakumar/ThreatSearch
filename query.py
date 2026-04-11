import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from nlp      import process_line
from ranker   import Ranker
from expander import QueryExpander


# ── Display ───────────────────────────────────────────────────────────────────

_CVE_BASE    = "https://nvd.nist.gov/vuln/detail/"
_ATTACK_BASE = "https://attack.mitre.org/techniques/"

def _url(doc_id: str) -> str:
    if doc_id.upper().startswith("CVE-"):
        return _CVE_BASE + doc_id.upper()
    if doc_id.upper().startswith("T") and doc_id[1:].replace(".", "").isdigit():
        tid = doc_id.upper().replace(".", "/")
        return _ATTACK_BASE + tid + "/"
    return ""

def _source_label(source: str) -> str:
    return {"nvd": "NVD  ", "attack": "ATT&CK"}.get(source, source)

def print_results(
    results:      list[tuple[str, float, str]],
    query_terms:  list[str],
    elapsed:      float,
    explain_text: str = "",
):
    print()
    if explain_text:
        print(explain_text)
        print()

    if not results:
        print("No results found.")
        return

    print(f"Query terms (processed): {query_terms}")
    print(f"Found {len(results)} result(s) in {elapsed*1000:.1f} ms\n")

    col_w = max(len(r[0]) for r in results) + 2
    header = f"{'Rank':<5}  {'Score':>7}  {'Source':<7}  {'Doc ID':<{col_w}}  URL"
    print(header)
    print("─" * len(header))

    for rank, (doc_id, score, source) in enumerate(results, 1):
        url = _url(doc_id)
        print(f"{rank:<5}  {score:>7.4f}  {_source_label(source):<7}  {doc_id:<{col_w}}  {url}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def search(
    query:            str,
    index_dir:        str  = "data/index",
    doc_lengths_path: str  = "data/index/doc_lengths.json",
    attack_path:      str  = "data/attack/enterprise-attack.json",
    nvd_map_path:     str  = "data/index/technique_cve_map.json",
    scorer:           str  = "bm25",
    top_n:            int  = 10,
    expand:           bool = True,
    source_weight:    bool = True,
    explain:          bool = False,
    boolean_mode:     bool = False,
) -> list[tuple[str, float, str]]:
    """
    Run a full ThreatSearch query.  Returns ranked (doc_id, score, source) list.
    Can be imported and called programmatically from evaluate.py.
    """
    # 1. Boolean pre-filter (if query contains AND / OR / NOT)
    import re as _re
    boolean_filter_ids = None
    is_boolean = boolean_mode or bool(_re.search(r"\b(AND|OR|NOT)\b", query, _re.IGNORECASE))
    if is_boolean:
        from boolean_query import BooleanQueryHandler
        bq = BooleanQueryHandler(index_dir)
        boolean_filter_ids = bq.execute(query)
        if not boolean_filter_ids:
            print("[query] Boolean filter matched 0 documents.", file=sys.stderr)
            return []
        print(f"[query] Boolean filter: {len(boolean_filter_ids):,} candidate docs", file=sys.stderr)

    # 2. NLP processing (strip Boolean operators before stemming)
    clean_query = _re.sub(r"\b(AND|OR|NOT)\b", " ", query, flags=_re.IGNORECASE)
    _, terms = process_line("Q " + clean_query)
    if not terms:
        print("[query] No indexable terms after NLP processing.", file=sys.stderr)
        return []

    # 2. Load components (done once per session in practice; here per call for simplicity)
    ranker = Ranker(
        index_dir    = index_dir,
        doc_lengths  = doc_lengths_path,
        scorer       = scorer,
        source_weight= source_weight,
    )

    expanded_terms = terms
    term_weights   = None
    explain_text   = ""

    if expand and os.path.exists(attack_path):
        expander = QueryExpander(
            attack_path = attack_path,
            nvd_mapping = nvd_map_path if os.path.exists(nvd_map_path) else None,
        )
        expanded_terms, term_weights = expander.expand(terms)
        if explain:
            explain_text = expander.explain(terms)

    # 3. Rank
    t0      = time.perf_counter()
    results = ranker.score(expanded_terms, top_n=top_n, term_weights=term_weights,
                           candidate_ids=boolean_filter_ids)
    elapsed = time.perf_counter() - t0

    print_results(results, terms, elapsed, explain_text)
    return results


# ── Session mode: keep ranker/expander alive across multiple queries ───────────

def interactive_session(args):
    """
    Load index once, then loop accepting queries from stdin.
    Much faster than re-loading for each query.
    """
    print("[query] Loading index (one-time) …", file=sys.stderr)
    ranker = Ranker(
        index_dir    = args.index_dir,
        doc_lengths  = args.doc_lengths,
        scorer       = args.scorer,
        source_weight= not args.no_source_weight,
    )

    expander = None
    if not args.no_expand and os.path.exists(args.attack):
        expander = QueryExpander(
            attack_path = args.attack,
            nvd_mapping = args.nvd_map if os.path.exists(args.nvd_map) else None,
        )

    print("\nThreatSearch ready. Type a query and press Enter. Ctrl-C to exit.\n")

    while True:
        try:
            raw = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not raw:
            continue

        _, terms = process_line("Q " + raw)
        if not terms:
            print("No indexable terms.\n")
            continue

        expanded_terms = terms
        term_weights   = None
        explain_text   = ""

        if expander:
            expanded_terms, term_weights = expander.expand(terms)
            if args.explain:
                explain_text = expander.explain(terms)

        t0      = time.perf_counter()
        results = ranker.score(expanded_terms, top_n=args.top, term_weights=term_weights)
        elapsed = time.perf_counter() - t0

        print_results(results, terms, elapsed, explain_text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ThreatSearch — CVE and ATT&CK ranked retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("query",             nargs="*", help="Query string (omit for interactive mode)")
    ap.add_argument("--index-dir",       default="data/index")
    ap.add_argument("--doc-lengths",     default="data/index/doc_lengths.json")
    ap.add_argument("--attack",          default="data/attack/enterprise-attack.json")
    ap.add_argument("--nvd-map",         default="data/index/technique_cve_map.json")
    ap.add_argument("--scorer",          default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("--top",             default=10, type=int)
    ap.add_argument("--no-expand",       action="store_true")
    ap.add_argument("--explain",         action="store_true")
    ap.add_argument("--no-source-weight",action="store_true")
    ap.add_argument("--boolean", action="store_true",
                    help="Force Boolean mode (auto-detected from AND/OR/NOT in query)")
    args = ap.parse_args()

    # Interactive mode if no query given
    if not args.query:
        interactive_session(args)
    else:
        search(
            query            = " ".join(args.query),
            index_dir        = args.index_dir,
            doc_lengths_path = args.doc_lengths,
            attack_path      = args.attack,
            nvd_map_path     = args.nvd_map,
            scorer           = args.scorer,
            top_n            = args.top,
            expand           = not args.no_expand,
            source_weight    = not args.no_source_weight,
            explain          = args.explain,
            boolean_mode     = getattr(args, 'boolean', False),
        )
