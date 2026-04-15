
import glob
import os
import re
import sys
from typing import Set

sys.path.insert(0, os.path.dirname(__file__))
from nlp import process_line


def _tokenize_term(term: str) -> list[str]:
    """
    Run a single Boolean operand through the NLP pipeline.
    Returns list of stemmed tokens (same as the index uses).
    """
    _, tokens = process_line("Q " + term.strip())
    return tokens


# ── Parser ────────────────────────────────────────────────────────────────────
# Grammar (right-recursive, no precedence — use parentheses for grouping):
#   query  := term (OP term)*
#   OP     := AND | OR | AND NOT
#   term   := '(' query ')' | word+

class _Token:
    """Lexer token types."""
    WORD = "WORD"
    AND  = "AND"
    OR   = "OR"
    NOT  = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF  = "EOF"


def _lex(query: str) -> list[tuple[str, str]]:
    """
    Tokenise a Boolean query string into (type, value) pairs.
    Handles: AND, OR, NOT (case-insensitive), parentheses, word tokens.
    """
    tokens = []
    # Split on whitespace and parens while keeping parens as tokens
    parts = re.split(r'(\s+|\(|\))', query.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        up = part.upper()
        if up == "AND":
            tokens.append((_Token.AND, "AND"))
        elif up == "OR":
            tokens.append((_Token.OR, "OR"))
        elif up == "NOT":
            tokens.append((_Token.NOT, "NOT"))
        elif part == "(":
            tokens.append((_Token.LPAREN, "("))
        elif part == ")":
            tokens.append((_Token.RPAREN, ")"))
        else:
            tokens.append((_Token.WORD, part))
    tokens.append((_Token.EOF, ""))
    return tokens


class _Parser:
    """
    Recursive descent parser for Boolean queries.
    Returns an AST as nested tuples:
        ("AND", left, right)
        ("OR",  left, right)
        ("NOT", operand)
        ("TERM", [stemmed_tokens])
    """

    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos    = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _consume(self, expected_type: str = None) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        if expected_type and tok[0] != expected_type:
            raise ValueError(
                f"Boolean parse error: expected {expected_type} but got {tok} "
                f"at position {self._pos}"
            )
        self._pos += 1
        return tok

    def parse(self):
        node = self._parse_or()
        if self._peek()[0] != _Token.EOF:
            raise ValueError(
                f"Unexpected token at position {self._pos}: {self._peek()}"
            )
        return node

    def _parse_or(self):
        left = self._parse_and()
        while self._peek()[0] == _Token.OR:
            self._consume(_Token.OR)
            right = self._parse_and()
            left  = ("OR", left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._peek()[0] == _Token.AND:
            self._consume(_Token.AND)
            # Support "AND NOT" as a single operator
            if self._peek()[0] == _Token.NOT:
                self._consume(_Token.NOT)
                right = self._parse_primary()
                left  = ("AND_NOT", left, right)
            else:
                right = self._parse_not()
                left  = ("AND", left, right)
        return left

    def _parse_not(self):
        if self._peek()[0] == _Token.NOT:
            self._consume(_Token.NOT)
            operand = self._parse_primary()
            return ("NOT", operand)
        return self._parse_primary()

    def _parse_primary(self):
        if self._peek()[0] == _Token.LPAREN:
            self._consume(_Token.LPAREN)
            node = self._parse_or()
            self._consume(_Token.RPAREN)
            return node

        # Collect consecutive WORD tokens as a single phrase term
        words = []
        while self._peek()[0] == _Token.WORD:
            words.append(self._consume(_Token.WORD)[1])

        if not words:
            raise ValueError(
                f"Expected a query term at position {self._pos}, got {self._peek()}"
            )

        tokens = _tokenize_term(" ".join(words))
        return ("TERM", tokens)


class BooleanQueryHandler:
    """
    Loads inverted index shards and executes Boolean queries.

    Parameters
    ----------
    index_dir : directory containing index_*.txt files and dictionary.txt
    """

    def __init__(self, index_dir: str = "data/index"):
        self._index_dir   = index_dir
        self._shard_files = sorted(
            glob.glob(os.path.join(index_dir, "index_*.txt"))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No index_*.txt files found in {index_dir}"
            )

        # Build word → word_code map from dictionary
        dict_path = os.path.join(index_dir, "dictionary.txt")
        self._word_code: dict[str, int] = {}
        with open(dict_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self._word_code[line.strip()] = i

        # Cache: word_code → set of doc_ids
        self._posting_cache: dict[int, set[str]] = {}

        # Universe: all known doc_ids (populated lazily on first NOT query)
        self._universe: set[str] | None = None

        print(
            f"[boolean] {len(self._shard_files)} shards, "
            f"{len(self._word_code):,} terms",
            file=sys.stderr,
        )


    def _load_posting_set(self, term: str) -> set[str]:
        """
        Return the set of doc_ids that contain this term.
        Merges across all shards; results cached.
        """
        wc = self._word_code.get(term)
        if wc is None:
            return set()

        if wc in self._posting_cache:
            return self._posting_cache[wc]

        doc_ids: set[str] = set()
        for shard_path in self._shard_files:
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        if int(parts[0]) != wc:
                            continue
                    except ValueError:
                        continue
                    # Extract doc_ids from postings: (doc_id,tf,src)
                    for posting in parts[3:]:
                        inner = posting.strip("()")
                        segs  = inner.rsplit(",", 2)
                        if segs:
                            doc_ids.add(segs[0])
                    break  # found in this shard

        self._posting_cache[wc] = doc_ids
        return doc_ids

    def _term_set(self, tokens: list[str]) -> set[str]:
        """
        Return the intersection of posting sets for all tokens in a phrase.
        (A multi-word term requires all words to appear in the document.)
        """
        if not tokens:
            return set()
        result = self._load_posting_set(tokens[0])
        for token in tokens[1:]:
            result = result & self._load_posting_set(token)
        return result

    def _get_universe(self) -> set[str]:
        """Return the set of all doc_ids in the index (for NOT operations)."""
        if self._universe is not None:
            return self._universe

        universe: set[str] = set()
        for shard_path in self._shard_files:
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    for posting in parts[3:]:
                        inner = posting.strip("()")
                        segs  = inner.rsplit(",", 2)
                        if segs:
                            universe.add(segs[0])
        self._universe = universe
        return universe


    def _evaluate(self, node) -> set[str]:
        kind = node[0]

        if kind == "TERM":
            return self._term_set(node[1])

        if kind == "AND":
            left  = self._evaluate(node[1])
            right = self._evaluate(node[2])
            return left & right

        if kind == "OR":
            left  = self._evaluate(node[1])
            right = self._evaluate(node[2])
            return left | right

        if kind == "NOT":
            operand = self._evaluate(node[1])
            return self._get_universe() - operand

        if kind == "AND_NOT":
            left    = self._evaluate(node[1])
            exclude = self._evaluate(node[2])
            return left - exclude

        raise ValueError(f"Unknown AST node type: {kind}")


    def execute(self, query: str) -> set[str]:
        """
        Parse and execute a Boolean query.

        Returns set of matching doc_ids. Empty set = no matches.

        Raises ValueError if the query cannot be parsed.
        """
        tokens = _lex(query)
        parser = _Parser(tokens)
        ast    = parser.parse()
        result = self._evaluate(ast)
        return result





if __name__ == "__main__":
    import argparse
    from ranker import Ranker

    ap = argparse.ArgumentParser(description="ThreatSearch Boolean query handler")
    ap.add_argument("query",          nargs="+", help="Boolean query string")
    ap.add_argument("--index-dir",    default="data/index")
    ap.add_argument("--doc-lengths",  default="data/index/doc_lengths.json")
    ap.add_argument("--top",          default=10, type=int)
    ap.add_argument("--no-rank",      action="store_true",
                    help="Print Boolean matches without BM25 re-ranking")
    args = ap.parse_args()

    query_str = " ".join(args.query)

    bq = BooleanQueryHandler(args.index_dir)


    print(f"Query: {query_str}")
    matched = bq.execute(query_str)
    print(f"Boolean filter matched: {len(matched):,} documents\n")

    if not matched:
        print("No documents match the Boolean filter.")
        sys.exit(0)

    if args.no_rank:
        print("Top matches (unranked):")
        for doc_id in sorted(matched)[:args.top]:
            print(f"  {doc_id}")
    else:
        # Re-rank with BM25
        _, terms = process_line("Q " + re.sub(r'\b(AND|OR|NOT)\b', '', query_str, flags=re.IGNORECASE))
        ranker   = Ranker(args.index_dir, args.doc_lengths, scorer="bm25")

        # Filter posting cache to only matched docs
        results = ranker.score(terms, top_n=args.top * 10)   # over-fetch then filter
        filtered = [(doc_id, score, src)
                    for doc_id, score, src in results
                    if doc_id in matched][:args.top]

        if not filtered:
            print("BM25 re-ranking found no results within the Boolean-filtered set.")
            print("Top Boolean matches (unranked):")
            for doc_id in sorted(matched)[:args.top]:
                print(f"  {doc_id}")
        else:
            print(f"{'Rank':<5}  {'Score':>7}  {'Source':<7}  Doc ID")
            print("─" * 50)
            for rank, (doc_id, score, source) in enumerate(filtered, 1):
                print(f"{rank:<5}  {score:>7.4f}  {source:<7}  {doc_id}")
