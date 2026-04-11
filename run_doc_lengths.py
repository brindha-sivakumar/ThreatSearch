import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nlp import process_line


def compute_doc_lengths(corpus_dir: str) -> dict[str, int]:
    shard_files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not shard_files:
        raise FileNotFoundError(f"No .txt shards found in {corpus_dir}")

    doc_lengths = {}
    for path in shard_files:
        print(f"  scanning {os.path.basename(path)} …", file=sys.stderr)
        with open(path, encoding="utf-8") as f:
            for line in f:
                doc_id, tokens = process_line(line)
                if doc_id:
                    doc_lengths[doc_id] = len(tokens)

    return doc_lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--out",        default="data/index/doc_lengths.json")
    args = ap.parse_args()

    print(f"Computing document lengths from {args.corpus_dir} …", file=sys.stderr)
    doc_lengths = compute_doc_lengths(args.corpus_dir)

    avg = sum(doc_lengths.values()) / max(len(doc_lengths), 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc_lengths, f)

    print(f"\n[doc_lengths] {len(doc_lengths):,} documents", file=sys.stderr)
    print(f"[doc_lengths] avg length : {avg:.1f} tokens",    file=sys.stderr)
    print(f"[doc_lengths] written    → {args.out}",          file=sys.stderr)


if __name__ == "__main__":
    main()
