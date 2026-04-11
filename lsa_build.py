

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ── Step 1: stream shards → sparse TF-IDF matrix ─────────────────────────────

def build_sparse_tfidf(index_dir: str, dictionary_path: str, doc_lengths_path: str):
    """
    Streams all index_*.txt shards and builds a sparse term-document TF-IDF matrix.
    Returns (tfidf_matrix, doc_ids, words) without ever materialising a dense array.

    Matrix shape: (V, N)  — V terms × N documents
    """
    print("[lsa_build] loading vocabulary …", file=sys.stderr)
    with open(dictionary_path, encoding="utf-8") as f:
        words = [line.strip() for line in f]
    V = len(words)

    print("[lsa_build] loading document registry …", file=sys.stderr)
    with open(doc_lengths_path, encoding="utf-8") as f:
        dl = json.load(f)
    doc_ids   = sorted(dl.keys())
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    N = len(doc_ids)
    print(f"[lsa_build] V={V:,} terms  N={N:,} docs", file=sys.stderr)

    # Accumulate raw TF values and per-term df for IDF calculation
    rows, cols, tf_data = [], [], []
    df: dict[int, int] = {}     # word_code → total df across all shards

    shard_files = sorted(glob.glob(os.path.join(index_dir, "index_*.txt")))
    if not shard_files:
        raise FileNotFoundError(f"No index_*.txt files in {index_dir}")

    for shard_path in shard_files:
        print(f"  streaming {os.path.basename(shard_path)} …", file=sys.stderr)
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    wc       = int(parts[0])
                    term_df  = int(parts[2])
                except ValueError:
                    continue

                # Accumulate df (a term may appear in multiple shards)
                df[wc] = df.get(wc, 0) + term_df

                for posting_str in parts[3:]:
                    inner = posting_str.strip("()")
                    segs  = inner.rsplit(",", 2)
                    if len(segs) != 3:
                        continue
                    doc_id, tf_str, _ = segs
                    col = doc_index.get(doc_id)
                    if col is None:
                        continue
                    try:
                        tf = int(tf_str)
                    except ValueError:
                        continue
                    rows.append(wc)
                    cols.append(col)
                    tf_data.append(float(tf))

    print(f"[lsa_build] {len(tf_data):,} non-zero entries collected", file=sys.stderr)

    # Build sparse TF matrix
    tf_matrix = sp.csr_matrix(
        (tf_data, (rows, cols)), shape=(V, N), dtype=np.float32
    )

    # Build IDF vector (BM25-style smoothed IDF, same as ranker.py)
    idf_vec = np.array([
        np.log((N - df.get(wc, 0) + 0.5) / (df.get(wc, 0) + 0.5) + 1)
        for wc in range(V)
    ], dtype=np.float32)

    # Scale rows by IDF (sparse diagonal multiply — never dense)
    print("[lsa_build] applying TF-IDF weighting …", file=sys.stderr)
    tfidf_matrix = sp.diags(idf_vec) @ tf_matrix   # still sparse

    return tfidf_matrix, doc_ids, words


# ── Step 2: truncated SVD ──────────────────────────────────────────────────────

def apply_svd(tfidf_matrix: sp.spmatrix, n_components: int = 300):
    """
    Apply TruncatedSVD to the TF-IDF matrix.
    Works entirely on the sparse matrix — no dense materialisation.

    Returns
    -------
    term_matrix  : (V, n_components)  float32 — each term as a latent vector
    doc_matrix   : (N, n_components)  float32 — each doc  as a latent vector
    explained_var: float — fraction of variance retained
    """
    print(
        f"[lsa_build] running TruncatedSVD  n_components={n_components} …",
        file=sys.stderr,
    )
    t0  = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=42)

    # fit on term-document matrix; transform gives (V, n_components)
    term_matrix = svd.fit_transform(tfidf_matrix)   # V × k
    doc_matrix  = svd.components_.T                 # N × k  (V^T · U · Σ)^T

    # L2-normalise rows so cosine similarity = dot product
    term_matrix = normalize(term_matrix, norm="l2")
    doc_matrix  = normalize(doc_matrix,  norm="l2")

    elapsed      = time.perf_counter() - t0
    explained    = svd.explained_variance_ratio_.sum()
    print(
        f"[lsa_build] SVD done in {elapsed:.1f}s  "
        f"variance retained: {explained*100:.1f}%",
        file=sys.stderr,
    )
    return term_matrix.astype(np.float32), doc_matrix.astype(np.float32), explained


# ── Step 3: save ──────────────────────────────────────────────────────────────

def save_lsa_index(
    term_matrix: np.ndarray,
    doc_matrix:  np.ndarray,
    doc_ids:     list[str],
    words:       list[str],
    out_dir:     str,
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "term_matrix.npy"), term_matrix)
    np.save(os.path.join(out_dir, "doc_matrix.npy"),  doc_matrix)
    with open(os.path.join(out_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    with open(os.path.join(out_dir, "words.json"), "w", encoding="utf-8") as f:
        json.dump(words, f)

    tm_mb = term_matrix.nbytes / 1e6
    dm_mb = doc_matrix.nbytes  / 1e6
    print(f"\n[lsa_build] saved to {out_dir}/", file=sys.stderr)
    print(f"  term_matrix.npy  shape={term_matrix.shape}  {tm_mb:.1f} MB", file=sys.stderr)
    print(f"  doc_matrix.npy   shape={doc_matrix.shape}   {dm_mb:.1f} MB", file=sys.stderr)
    print(f"  total on disk    {tm_mb + dm_mb:.1f} MB", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def build(
    index_dir:        str = "data/index",
    dictionary_path:  str = "data/index/dictionary.txt",
    doc_lengths_path: str = "data/index/doc_lengths.json",
    out_dir:          str = "data/lsa_index",
    n_components:     int = 300,
):
    t_total = time.perf_counter()

    tfidf, doc_ids, words = build_sparse_tfidf(
        index_dir, dictionary_path, doc_lengths_path
    )
    term_matrix, doc_matrix, explained = apply_svd(tfidf, n_components)
    save_lsa_index(term_matrix, doc_matrix, doc_ids, words, out_dir)

    print(
        f"\n[lsa_build] complete in {time.perf_counter()-t_total:.1f}s  "
        f"variance retained: {explained*100:.1f}%",
        file=sys.stderr,
    )
    return term_matrix, doc_matrix, doc_ids, words


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ThreatSearch LSA index builder")
    ap.add_argument("--index-dir",   default="data/index")
    ap.add_argument("--dict",        default="data/index/dictionary.txt")
    ap.add_argument("--doc-lengths", default="data/index/doc_lengths.json")
    ap.add_argument("--out-dir",     default="data/lsa_index")
    ap.add_argument("--components",  default=300, type=int,
                    help="Latent dimensions (default 300; use 100 for a quick test)")
    args = ap.parse_args()

    build(
        index_dir       = args.index_dir,
        dictionary_path = args.dict,
        doc_lengths_path= args.doc_lengths,
        out_dir         = args.out_dir,
        n_components    = args.components,
    )
