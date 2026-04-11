import re
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# ── Stemmer ───────────────────────────────────────────────────────────────────
_ps = PorterStemmer()

# ── Stopwords ─────────────────────────────────────────────────────────────────
# Start from NLTK English stopwords, then add security-domain terms that appear
# in almost every CVE or ATT&CK description and carry no discriminating signal.
_BASE_STOPS = set(stopwords.words("english"))

_SECURITY_STOPS = {
    # Near-universal in CVE descriptions
    "vulnerability", "vulnerabilities", "vulnerable", "issue", "issues",
    "advisory", "advisories", "update", "updated", "updates",
    "security", "securely", "allow", "allows", "allowed", "allowing",
    "attacker", "attackers", "attack", "attacks",
    "version", "versions", "affect", "affected", "affecting", "affects",
    "product", "products", "system", "systems", "user", "users",
    "remote", "local", "arbitrary", "execute", "execution",
    # Near-universal in ATT&CK descriptions
    "adversary", "adversaries", "technique", "techniques",
    "may", "use", "used", "using", "via", "also", "well",
    "example", "examples", "following", "specific", "various",
    # Generic English words that survive NLTK but add no value here
    "however", "therefore", "thus", "although", "whether",
    "within", "without", "across", "further",
}

STOP_WORDS = _BASE_STOPS | _SECURITY_STOPS

# ── Structured ID pattern (must NOT be stemmed or alpha-stripped) ─────────────
_STRUCTURED_ID = re.compile(
    r'^(CVE-\d{4}-\d+|CWE-\d+|CAPEC-\d+|T\d{4}(?:\.\d{3})?|TA\d{4}|MS\d{2}-\d+)$',
    re.IGNORECASE
)


def is_structured_id(token: str) -> bool:
    return bool(_STRUCTURED_ID.match(token))


# ── Token cleaning (from checkpoint 1 parser, extended) ────────────────────────────────

def clean_token(token: str) -> str | None:
    """
    Clean a single whitespace-split token.
    Returns a normalised string, or None if the token should be dropped.
    Structured IDs bypass all cleaning and are returned uppercased.
    """
    # Preserve structured security identifiers exactly
    if is_structured_id(token):
        return token.upper()

    # Strip URLs
    token = re.sub(r'https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|uk|io)\S*', '', token)

    # Strip HTML entities and tags
    token = re.sub(r'&[a-z]+;', '', token)
    token = re.sub(r'<.*?>', '', token)
    token = re.sub(r'lt;.*?gt;?', '', token)
    token = re.sub(r'</?\w+gt', '', token)

    # Strip special characters
    token = re.sub(r'[@#$<>\[\]_]', '', token)
    token = token.strip(string.punctuation)

    if not token:
        return None

    # Tokens with digits are dropped (except structured IDs already handled)
    if any(c.isdigit() for c in token):
        return None

    # Keep only alphabetic characters
    alpha = re.sub(r'[^a-zA-Z]', '', token)

    # Minimum meaningful length
    if len(alpha) <= 2:
        return None

    return alpha.lower()


# ── Full token pipeline ───────────────────────────────────────────────────────

def process_tokens(raw_tokens: list[str]) -> list[str]:
    """
    Clean → filter stopwords → stem.
    Structured IDs are returned as-is (uppercased, not stemmed).
    Returns list of final tokens for indexing.
    """
    result = []
    for raw in raw_tokens:
        cleaned = clean_token(raw)
        if cleaned is None:
            continue
        # Structured IDs: not stemmed, not stopword-filtered
        if is_structured_id(cleaned):
            result.append(cleaned)
            continue
        # Regular tokens: stopword filter then stem
        if cleaned in STOP_WORDS:
            continue
        stemmed = _ps.stem(cleaned)
        if stemmed and len(stemmed) > 2:
            result.append(stemmed)
    return result


# ── Line-level entry point (used by index.py) ─────────────────────────────────

def process_line(line: str) -> tuple[str, list[str]]:
    """
    Parse one corpus line (output of ingest.py):
        '<doc-id> <token1> <token2> ...'

    Returns (doc_id, processed_tokens).
    doc_id is the raw identifier (e.g. 'CVE-2021-44228', 'T1021.002').
    """
    parts = line.strip().split()
    if not parts:
        return "", []
    doc_id = parts[0]
    raw_tokens = parts[1:]
    return doc_id, process_tokens(raw_tokens)


# ── Standalone inspection ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python nlp.py <corpus_shard.txt>")
        print("       Prints (doc_id, token_count, first 10 tokens) for each line.")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc_id, tokens = process_line(line)
            if not doc_id:
                continue
            print(f"{doc_id:30s}  n={len(tokens):4d}  {tokens[:10]}")
            if i >= 19:
                print("... (showing first 20 docs)")
                break
