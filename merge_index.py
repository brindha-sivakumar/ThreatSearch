import argparse
import glob
import heapq
import os
import sys
import time


def parse_index_line(line: str) -> tuple[int, str, int, str] | None:
    """
    Parse one line from an index shard file.

    Format:  <word-code> <word> <df> (<doc-id>,<tf>,<src>) ...

    Returns (word_code, word, df, postings_str) or None if malformed.
    postings_str is the raw space-joined posting tokens — kept as a
    string to avoid re-serialising individual postings.
    """
    parts = line.rstrip("\n").split()
    if len(parts) < 4:
        return None
    try:
        wc = int(parts[0])
    except ValueError:
        return None

    word    = parts[1]
    try:
        df  = int(parts[2])
    except ValueError:
        return None

    postings_str = " ".join(parts[3:])
    return wc, word, df, postings_str



class ShardReader:
    """
    Wraps one open shard file.  Reads one line at a time via next().
    Implements __lt__ so instances can be pushed onto heapq by word_code.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh  = open(path, encoding="utf-8")
        self._cur  = None          # current parsed line tuple
        self._advance()

    def _advance(self):
        """Read and parse the next non-empty line.  Sets _cur to None at EOF."""
        while True:
            raw = self._fh.readline()
            if not raw:
                self._cur = None
                self._fh.close()
                return
            parsed = parse_index_line(raw)
            if parsed is not None:
                self._cur = parsed   # (word_code, word, df, postings_str)
                return

    @property
    def exhausted(self) -> bool:
        return self._cur is None

    @property
    def word_code(self) -> int:
        return self._cur[0]

    def pop(self) -> tuple[int, str, int, str]:
        """Return current line tuple and advance to the next."""
        result = self._cur
        self._advance()
        return result

    def __lt__(self, other: "ShardReader") -> bool:
        """heapq comparison — sort by word_code ascending."""
        return self.word_code < other.word_code


def kway_merge(
    shard_files: list[str],
    out_path:    str,
):
    """
    K-way merge of sorted shard files into a single merged index file.
    """

    readers = []
    for path in shard_files:
        r = ShardReader(path)
        if not r.exhausted:
            heapq.heappush(readers, r)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.perf_counter()

    terms_written  = 0
    total_postings = 0

    with open(out_path, "w", encoding="utf-8") as out:
        while readers:
            
            min_wc = readers[0].word_code

            same_wc_readers = []
            while readers and readers[0].word_code == min_wc:
                same_wc_readers.append(heapq.heappop(readers))

            word          = same_wc_readers[0]._cur[1]
            merged_df     = 0
            posting_parts = []

            for r in same_wc_readers:
                wc, _word, df, postings_str = r.pop()
                merged_df     += df
                posting_parts.append(postings_str)

                # Re-push reader if it has more lines
                if not r.exhausted:
                    heapq.heappush(readers, r)

            merged_postings = " ".join(posting_parts)
            out.write(f"{min_wc} {word} {merged_df} {merged_postings}\n")

            terms_written  += 1
            total_postings += merged_df

            if terms_written % 10_000 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  {terms_written:,} terms written  "
                    f"{total_postings:,} total postings  "
                    f"({elapsed:.1f}s)",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(out_path) / 1e6

    
    print(
        f"\n[merge] done in {elapsed:.1f}s\n"
        f"  terms      : {terms_written:,}\n"
        f"  postings   : {total_postings:,}\n"
        f"  output     : {out_path}  ({size_mb:.1f} MB)",
        file=sys.stderr,
    )

    return terms_written, total_postings




def merge(
    index_dir:   str  = "data/index",
    out_path:    str  = None,
    keep_shards: bool = False,
    verbose:     bool = True,
):
    shard_files = sorted(
        glob.glob(os.path.join(index_dir, "index_*.txt"))
    )

    # Exclude any previously merged file from the input list
    shard_files = [p for p in shard_files if "merged" not in os.path.basename(p)]

    if not shard_files:
        raise FileNotFoundError(f"No index_*.txt shard files found in {index_dir}")

    if out_path is None:
        out_path = os.path.join(index_dir, "index_merged.txt")

    if verbose:
        print(f"[merge] {len(shard_files)} shard file(s) → {out_path}", file=sys.stderr)
        for p in shard_files:
            print(f"  {os.path.basename(p)}", file=sys.stderr)
        print(file=sys.stderr)

    kway_merge(shard_files, out_path)


    if not keep_shards:
        print("\n[merge] removing shard files …", file=sys.stderr)
        for p in shard_files:
            os.remove(p)
            print(f"  deleted {os.path.basename(p)}", file=sys.stderr)

   
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="ThreatSearch — merge per-shard index files into one global index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--index-dir",    default="data/index",
                    help="Directory containing index_*.txt shard files")
    ap.add_argument("--out",          default=None,
                    help="Output path (default: <index-dir>/index_merged.txt)")
    ap.add_argument("--keep-shards",  action="store_true",
                    help="Keep individual shard files after merging")
    ap.add_argument("--quiet",        action="store_true",
                    help="Suppress progress output")
    args = ap.parse_args()

    merge(
        index_dir   = args.index_dir,
        out_path    = args.out,
        keep_shards = args.keep_shards,
        verbose     = not args.quiet,
    )