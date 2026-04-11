
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import normalize


class LSARanker:
    """
    Memory-safe ranker backed by pre-built LSA matrices.

    Parameters
    ----------
    lsa_dir       : directory containing term_matrix.npy, doc_matrix.npy,
                    doc_ids.json, words.json (output of lsa_build.py)
    dictionary_path: path to dictionary.txt (used to resolve query term → word_code)
    """

    def __init__(self, lsa_dir: str = "data/lsa_index",
                 dictionary_path: str = "data/index/dictionary.txt"):

        print("[lsa_ranker] loading LSA index …", file=sys.stderr)
        t0 = time.perf_counter()

        # Load fixed-size matrices
        self.term_matrix = np.load(
            os.path.join(lsa_dir, "term_matrix.npy")
        )   # shape: (V, k)
        self.doc_matrix  = np.load(
            os.path.join(lsa_dir, "doc_matrix.npy")
        )   # shape: (N, k)

        with open(os.path.join(lsa_dir, "doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        with open(os.path.join(lsa_dir, "words.json"), encoding="utf-8") as f:
            words = json.load(f)

        # Build word → row-index lookup (same order as dictionary.txt)
        self.word_code: dict[str, int] = {w: i for i, w in enumerate(words)}

        elapsed = time.perf_counter() - t0
        tm_mb   = self.term_matrix.nbytes / 1e6
        dm_mb   = self.doc_matrix.nbytes  / 1e6

        print(
            f"[lsa_ranker] ready in {elapsed:.2f}s  "
            f"V={self.term_matrix.shape[0]:,}  "
            f"N={self.doc_matrix.shape[0]:,}  "
            f"k={self.term_matrix.shape[1]}  "
            f"mem={tm_mb+dm_mb:.0f} MB (fixed)",
            file=sys.stderr,
        )

    # ── Core query ────────────────────────────────────────────────────────────

    def query(
        self,
        query_terms:  list[str],
        top_n:        int   = 10,
        term_weights: dict  = None,
    ) -> list[tuple[str, float, str]]:
        """
        Project query into latent space, return top-N docs by cosine similarity.

        Parameters
        ----------
        query_terms  : pre-processed (stemmed, filtered) query tokens
        top_n        : number of results
        term_weights : optional {term: weight} from expander (same interface as BM25)

        Returns
        -------
        List of (doc_id, score, source) — source inferred from doc_id prefix.
        Scores are cosine similarities in [−1, 1]; typically 0.1–0.6 for matches.
        """
        if term_weights is None:
            term_weights = {}

        # Build query vector as weighted average of term latent vectors
        vecs, weights = [], []
        for term in query_terms:
            wc = self.word_code.get(term)
            if wc is None:
                continue
            vecs.append(self.term_matrix[wc])
            weights.append(term_weights.get(term, 1.0))

        if not vecs:
            return []

        weights    = np.array(weights, dtype=np.float32)
        vecs       = np.stack(vecs)                         # (n_terms, k)
        query_vec  = (weights[:, None] * vecs).sum(axis=0) # weighted sum
        query_vec  = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        # Cosine similarity: doc_matrix rows already L2-normalised → dot product
        scores    = self.doc_matrix @ query_vec             # (N,)
        top_idx   = np.argpartition(scores, -top_n)[-top_n:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            doc_id = self.doc_ids[idx]
            score  = float(scores[idx])
            source = self._infer_source(doc_id)
            results.append((doc_id, score, source))

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_source(doc_id: str) -> str:
        """Guess source tag from the doc_id format."""
        uid = doc_id.upper()
        if uid.startswith("CVE-"):
            return "nvd"
        if uid.startswith("T") and uid[1:5].isdigit():
            return "attack"
        return "unknown"

    def memory_usage_mb(self) -> float:
        """Return total RAM used by the two matrices in MB."""
        return (self.term_matrix.nbytes + self.doc_matrix.nbytes) / 1e6

    def explain_query(self, query_terms: list[str]) -> str:
        """
        Show which query terms were found in the vocabulary and their
        cosine similarity to each other (useful for debugging semantic drift).
        """
        lines = ["LSA query explanation:"]
        found, missing = [], []
        for t in query_terms:
            if t in self.word_code:
                found.append(t)
            else:
                missing.append(t)
        lines.append(f"  found in vocabulary : {found}")
        if missing:
            lines.append(f"  not in vocabulary   : {missing}")

        if len(found) >= 2:
            lines.append("  pairwise cosine similarity between query terms:")
            for i in range(len(found)):
                for j in range(i + 1, len(found)):
                    wi = self.word_code[found[i]]
                    wj = self.word_code[found[j]]
                    sim = float(self.term_matrix[wi] @ self.term_matrix[wj])
                    lines.append(f"    {found[i]} ↔ {found[j]} : {sim:.3f}")
        return "\n".join(lines)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from nlp import process_line

    ap = argparse.ArgumentParser(description="LSA ranker smoke-test")
    ap.add_argument("--lsa-dir",   default="data/lsa_index")
    ap.add_argument("--dict",      default="data/index/dictionary.txt")
    ap.add_argument("--top",       default=10, type=int)
    ap.add_argument("--explain",   action="store_true")
    ap.add_argument("query",       nargs="+")
    args = ap.parse_args()

    ranker = LSARanker(args.lsa_dir, args.dict)
    print(f"[lsa_ranker] memory footprint: {ranker.memory_usage_mb():.0f} MB (fixed)\n")

    _, terms = process_line("Q " + " ".join(args.query))
    if not terms:
        print("No indexable terms after NLP processing.")
        sys.exit(0)

    if args.explain:
        print(ranker.explain_query(terms))
        print()

    t0      = time.perf_counter()
    results = ranker.query(terms, top_n=args.top)
    elapsed = time.perf_counter() - t0

    print(f"Query terms (processed): {terms}")
    print(f"Scorer: LSA cosine similarity")
    print(f"Time  : {elapsed*1000:.1f} ms\n")

    if not results:
        print("No results found.")
    else:
        col_w = max(len(r[0]) for r in results) + 2
        print(f"{'Rank':<5} {'Score':>7}  {'Source':<7}  Doc ID")
        print("─" * 50)
        for rank, (doc_id, score, source) in enumerate(results, 1):
            print(f"{rank:<5} {score:>7.4f}  {source:<7}  {doc_id}")
