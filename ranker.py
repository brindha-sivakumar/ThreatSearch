
import glob
import json
import math
import os
import sys
from collections import defaultdict

# ── Constants ─────────────────────────────────────────────────────────────────
BM25_K1 = 1.5
BM25_B  = 0.75
SOURCE_WEIGHTS = {"attack": 1.3, "nvd": 1.0}   # ATT&CK postings get a boost


class Ranker:
    """
    Loads dictionary + index shards from index_dir and scores queries.

    Parameters
    ----------
    index_dir    : directory produced by index.py (contains dictionary.txt + index_*.txt)
    doc_lengths  : path to doc_lengths.json produced by run_doc_lengths.py
    scorer       : "bm25" (default) or "tfidf"
    source_weight: whether to apply ATT&CK source boost
    """

    def __init__(
        self,
        index_dir:     str  = "data/index",
        doc_lengths:   str  = "data/index/doc_lengths.json",
        scorer:        str  = "bm25",
        source_weight: bool = True,
    ):
        self.index_dir     = index_dir
        self.scorer        = scorer
        self.source_weight = source_weight

        print("[ranker] loading dictionary", file=sys.stderr)
        self.word_code, self.code_word = self._load_dictionary()

        print("[ranker] loading document lengths", file=sys.stderr)
        self.doc_lengths, self.avg_dl, self.N = self._load_doc_lengths(doc_lengths)

        self.merged_index = sorted(glob.glob(os.path.join(index_dir, "index_*.txt")))
        if not self.merged_index:
            raise FileNotFoundError(f"No index_*.txt files found in {index_dir}")

        self._posting_cache: dict[int, dict[str, tuple[int, str]]] = {}

        # df cache: word_code → document frequency (summed across all shards)
        self._df_cache: dict[int, int] = {}

        print(
            f"[ranker] ready — {len(self.word_code):,} terms, "
            f"{self.N:,} docs, {len(self.merged_index)} shards, scorer={scorer}",
            file=sys.stderr,
        )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_dictionary(self) -> tuple[dict[str, int], dict[int, str]]:
        path = os.path.join(self.index_dir, "dictionary.txt")
        word_code, code_word = {}, {}
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                w = line.strip()
                word_code[w] = i
                code_word[i] = w
        return word_code, code_word

    def _load_doc_lengths(self, path: str) -> tuple[dict[str, int], float, int]:
        with open(path, encoding="utf-8") as f:
            dl = json.load(f)
        n      = len(dl)
        avg_dl = sum(dl.values()) / max(n, 1)
        return dl, avg_dl, n

    # ── Posting list access ───────────────────────────────────────────────────

    def _load_postings_for_term(self, term: str):
        """
        Find and cache the posting list for a single term.
        Reads every shard once per term, merging postings across shards.
        Results are stored in self._posting_cache[wc] and self._df_cache[wc].
        """
        wc = self.word_code.get(term)
        if wc is None:
            return   # term not in vocabulary
        if wc in self._posting_cache:
            return   # already loaded

        merged_postings: dict[str, tuple[int, str]] = {}
        total_df = 0

        
        with open(self.merged_index, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if not parts or int(parts[0]) != wc:
                    continue
                for posting_str in parts[3:]:
                    inner = posting_str.strip("()")
                    segments = inner.rsplit(",", 2)
                    if len(segments) != 3:
                        continue
                    doc_id, tf_str, src = segments
                    try:
                        tf = int(tf_str)
                    except ValueError:
                        continue
                    merged_postings[doc_id] = (tf, src)
                break   # found the term in this shard, move to next shard

        self._posting_cache[wc] = merged_postings
        self._df_cache[wc]      = len(merged_postings)

    def get_postings(self, term: str) -> dict[str, tuple[int, str]]:
        """Return {doc_id: (tf, source)} for a term. Empty dict if unknown."""
        self._load_postings_for_term(term)
        wc = self.word_code.get(term)
        if wc is None:
            return {}
        return self._posting_cache.get(wc, {})

    def get_df(self, term: str) -> int:
        """Return document frequency for a term across all shards."""
        self._load_postings_for_term(term)
        wc = self.word_code.get(term)
        return self._df_cache.get(wc, 0)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _idf(self, df: int) -> float:
        """Robertson-Sparck Jones IDF with smoothing."""
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _bm25_term_score(self, tf: int, df: int, dl: int) -> float:
        idf = self._idf(df)
        tf_norm = (tf * (BM25_K1 + 1)) / (
            tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(self.avg_dl, 1))
        )
        return idf * tf_norm

    def _tfidf_term_score(self, tf: int, df: int) -> float:
        return (1 + math.log(tf)) * self._idf(df)

    def score(
        self,
        query_terms: list[str],
        top_n: int = 10,
        term_weights: dict[str, float] | None = None,
        candidate_ids: set[str] | None = None,
    ) -> list[tuple[str, float, str]]:
        """
        Score all documents against query_terms.

        Parameters
        ----------
        query_terms  : pre-processed (stemmed, filtered) query tokens
        top_n        : number of results to return
        term_weights : optional per-term weight multiplier (used by expander)

        Returns
        -------
        List of (doc_id, score, source) sorted by descending score.
        """
        if term_weights is None:
            term_weights = {}

        doc_scores: dict[str, float] = defaultdict(float)
        doc_source: dict[str, str]   = {}

        for term in set(query_terms):   # deduplicate
            postings = self.get_postings(term)
            if not postings:
                continue

            df    = self.get_df(term)
            tw    = term_weights.get(term, 1.0)

            for doc_id, (tf, source) in postings.items():
                if candidate_ids is not None and doc_id not in candidate_ids:
                    continue
                dl = self.doc_lengths.get(doc_id, int(self.avg_dl))

                if self.scorer == "bm25":
                    term_score = self._bm25_term_score(tf, df, dl)
                else:
                    term_score = self._tfidf_term_score(tf, df)

                src_w = SOURCE_WEIGHTS.get(source, 1.0) if self.source_weight else 1.0
                doc_scores[doc_id] += term_score * tw * src_w
                doc_source[doc_id]  = source

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_source[doc_id]) for doc_id, score in ranked[:top_n]]


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from nlp import process_line

    ap = argparse.ArgumentParser(description="Quick ranker smoke-test")
    ap.add_argument("--index-dir",    default="data/index")
    ap.add_argument("--doc-lengths",  default="data/index/doc_lengths.json")
    ap.add_argument("--scorer",       default="bm25", choices=["bm25", "tfidf"])
    ap.add_argument("query", nargs="+", help="Query terms (will be NLP-processed)")
    args = ap.parse_args()

    ranker = Ranker(args.index_dir, args.doc_lengths, scorer=args.scorer)

    # Run through NLP pipeline
    raw_line = "QUERY " + " ".join(args.query)
    _, terms = process_line(raw_line)

    if not terms:
        print("No indexable terms after NLP processing.")
        sys.exit(0)

    print(f"\nQuery terms (processed): {terms}")
    print(f"Scorer: {args.scorer}\n")

    results = ranker.score(terms, top_n=10)
    if not results:
        print("No results found.")
    else:
        print(f"{'Rank':<5} {'Score':>8}  {'Source':<8}  Doc ID")
        print("─" * 50)
        for rank, (doc_id, score, source) in enumerate(results, 1):
            print(f"{rank:<5} {score:>8.4f}  {source:<8}  {doc_id}")
