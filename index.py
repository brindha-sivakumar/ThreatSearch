import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict

from nlp import process_line   # ThreatSearch security-aware NLP layer


def source_tag(filename: str) -> str:
    """Infer 'nvd' or 'attack' from the shard filename."""
    name = os.path.basename(filename).lower()
    if "attack" in name:
        return "attack"
    return "nvd"


def build_vocabulary(shard_files: list[str]) -> tuple[list[str], dict[str, int]]:
    """
    Stream every shard once.  Collect all processed terms into a set.
    Return (sorted_words, word_code_map).

    word_code_map[word] = line index in dictionary.txt (0-based).
    """
    vocab: set[str] = set()

    for path in shard_files:
        print(f"[index pass1] {os.path.basename(path)}", file=sys.stderr)
        with open(path, encoding="utf-8") as f:
            for line in f:
                _, tokens = process_line(line)
                vocab.update(tokens)

    sorted_words = sorted(vocab)
    word_code = {w: i for i, w in enumerate(sorted_words)}
    print(f"[index pass1] vocabulary size: {len(sorted_words):,}", file=sys.stderr)
    return sorted_words, word_code


def write_dictionary(sorted_words: list[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for w in sorted_words:
            f.write(w + "\n")
    print(f"[index] dictionary → {path}  ({len(sorted_words):,} terms)", file=sys.stderr)


def build_local_index(
    shard_path: str,
    word_code: dict[str, int],
    source: str
) -> dict[int, dict]:
    """
    Stream one shard.  Return a local index:
        { word_code: { word, df, postings: {doc_id: tf} } }

    Only terms with actual postings in this shard appear in the result —
    the index is intentionally sparse relative to the global dictionary.
    """
    index: dict[int, dict] = {}

    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            doc_id, tokens = process_line(line)
            if not doc_id or not tokens:
                continue

            tf_counts = Counter(tokens)
            seen_this_doc: set[int] = set()

            for term, tf in tf_counts.items():
                wc = word_code.get(term)
                if wc is None:
                    continue   # should not happen after pass 1

                if wc not in index:
                    index[wc] = {"word": term, "df": 0, "postings": {}}

                entry = index[wc]
                # Each doc_id is unique per line, so df += 1 is safe
                if wc not in seen_this_doc:
                    entry["df"] += 1
                    seen_this_doc.add(wc)

                # Store posting as (tf, source) — source distinguishes nvd vs attack
                entry["postings"][doc_id] = (tf, source)

    return index


def write_local_index(index: dict, out_path: str):
    """
    Write one shard's local index, sorted by word-code.

    Line format:
        <word-code> <word> <doc-frequency> (<doc-id>, <tf>, <source>) ...
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for wc in sorted(index):
            entry = index[wc]
            word = entry["word"]
            df   = entry["df"]
            postings_str = " ".join(
                f"({doc_id},{tf},{src})"
                for doc_id, (tf, src) in entry["postings"].items()
            )
            f.write(f"{wc} {word} {df} {postings_str}\n")


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(shard_files: list[str], index_files: list[str], dictionary_path: str):
    with open(dictionary_path) as f:
        vocab_size = sum(1 for _ in f)

    print("\n── Index summary ────────────────────────────────────")
    print(f"  Corpus shards  : {len(shard_files)}")
    print(f"  Index files    : {len(index_files)}")
    print(f"  Vocabulary     : {vocab_size:,} terms")

    total_postings = 0
    for path in index_files:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    total_postings += int(parts[2])   # df is the 3rd field
    print(f"  Total postings : {total_postings:,}")
    print("─────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index(corpus_dir: str, out_dir: str):
    shard_files = sorted(
        glob.glob(os.path.join(corpus_dir, "*.txt"))
    )
    if not shard_files:
        raise FileNotFoundError(f"No .txt shard files found in {corpus_dir}")

    print(f"[index] {len(shard_files)} shard(s) found", file=sys.stderr)
    os.makedirs(out_dir, exist_ok=True)

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    print("\n[index] Pass 1: building global vocabulary …", file=sys.stderr)
    sorted_words, word_code = build_vocabulary(shard_files)
    dict_path = os.path.join(out_dir, "dictionary.txt")
    write_dictionary(sorted_words, dict_path)

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    print("\n[index] Pass 2: building per-shard inverted indexes …", file=sys.stderr)
    index_files = []
    for shard_path in shard_files:
        source = source_tag(shard_path)
        local_index = build_local_index(shard_path, word_code, source)

        # Mirror the shard filename, e.g. nvd_0003.txt → index_nvd_0003.txt
        base = "index_" + os.path.basename(shard_path)
        out_path = os.path.join(out_dir, base)
        write_local_index(local_index, out_path)
        index_files.append(out_path)

        print(
            f"  {os.path.basename(shard_path):30s} → "
            f"{os.path.basename(out_path)}  ({len(local_index):,} terms)",
            file=sys.stderr,
        )


    print_summary(shard_files, index_files, dict_path)
    return dict_path, index_files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ThreatSearch index builder")
    ap.add_argument("--corpus-dir", default="data/corpus", help="Directory of corpus shards from ingest.py")
    ap.add_argument("--out-dir",    default="data/index",  help="Output directory for dictionary + index files")
    ap.add_argument("--no-verify",  action="store_true",   help="Skip word-code consistency check")
    args = ap.parse_args()

    build_index(args.corpus_dir, args.out_dir)
