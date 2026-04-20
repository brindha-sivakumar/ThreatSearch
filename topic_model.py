import argparse
import glob
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from nlp import process_line

# ── Topic-model-specific NLP ──────────────────────────────────────────────────
# Separate from the index NLP pipeline intentionally:
#   - No stemming        → readable topic words ("injection" not "inject")
#   - No structured IDs  → CVE-XXX / CWE-XXX add no topical meaning
#   - Aggressive extras  → boilerplate CVE words that survive standard stopwords
#   - Min word length 4  → drops "fix", "due", "set", "call" etc.

import re as _re
import string as _string

_TOPIC_EXTRA_STOPS = {
    # CVE boilerplate that survives standard stopwords
    "fix", "fixed", "fixes", "patch", "patched", "call", "called",
    "make", "makes", "made", "lead", "leads", "cause", "causes",
    "allow", "allows", "allowed", "could", "would", "should",
    "prior", "need", "needs", "check", "checks", "due",
    "set", "sets", "get", "gets", "use", "used", "uses",
    "include", "includes", "contain", "contains", "provide",
    "none", "note", "known", "public", "record", "number",
    "result", "issue", "state", "type", "name", "list",
    "version", "update", "exist", "occur", "happen",
    "early", "late", "base", "refer", "reject", "resolve",
    "reserve", "common", "possible", "reason", "require",
    "application", "component", "function", "endpoint",
    "request", "response", "data", "information", "access",
    # Expanded acronyms that become long unreadable strings
    "commonvulnerabilityexposure", "remotecodeexecution",
    "localprivilegeescalation", "denialofservice",
    "crosssitescripting", "sqlinjection",
}

# Structured ID pattern — skip these in topic NLP
_TOPIC_ID_PAT = _re.compile(
    r"^(CVE-\d{4}-\d+|CWE-\d+|T\d{4}(\.\d{3})?|TA\d{4})$",
    _re.IGNORECASE,
)

def topic_tokenize(text: str) -> list[str]:
    """
    NLP pipeline tailored for LDA topic modeling:
    - NO stemming   → preserves readable word forms
    - NO structured IDs   → CVE/CWE/ATT&CK IDs excluded
    - Aggressive stopword list including CVE boilerplate
    - Minimum 4-character alphabetic tokens only
    """
    try:
        from nltk.corpus import stopwords as _sw
        base_stops = set(_sw.words("english"))
    except Exception:
        base_stops = set()

    all_stops = base_stops | _TOPIC_EXTRA_STOPS

    # Strip HTML/markup
    text = _re.sub(r"<.*?>", " ", text)
    text = _re.sub(r"&[a-z]+;", " ", text)
    text = _re.sub(r"https?://\S+", " ", text)

    tokens = []
    for token in text.lower().split():
        # Drop structured IDs
        if _TOPIC_ID_PAT.match(token):
            continue
        # Keep only alphabetic characters
        alpha = _re.sub(r"[^a-z]", "", token)
        # Minimum length 4, not a stopword
        if len(alpha) >= 4 and alpha not in all_stops:
            tokens.append(alpha)

    return tokens


def stream_corpus(corpus_dir: str):
    """
    Yield (doc_id, raw_text) for every NVD document in corpus shards.
    ATT&CK technique documents are excluded — LDA is most meaningful
    over the larger, more varied CVE description corpus.
    Raw text is returned so topic_tokenize() (not the index NLP) is applied.
    """
    shard_files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not shard_files:
        raise FileNotFoundError(f"No .txt shards found in {corpus_dir}")

    for path in shard_files:
        # Skip ATT&CK shards for LDA
        if "attack" in os.path.basename(path).lower():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                doc_id   = parts[0]
                raw_text = parts[1]
                yield doc_id, raw_text


def load_corpus_for_lda(corpus_dir: str, min_doc_len: int = 5):
    """
    Load CVE corpus for LDA using topic_tokenize() — no stemming, no IDs,
    readable word forms for interpretable topic output.

    min_doc_len: skip documents with fewer than this many tokens
    """
    print("[lda] loading corpus with topic NLP (no stemming, no IDs) …", file=sys.stderr)
    doc_ids, texts = [], []
    for doc_id, raw_text in stream_corpus(corpus_dir):
        tokens = topic_tokenize(raw_text)
        if len(tokens) >= min_doc_len:
            doc_ids.append(doc_id)
            texts.append(tokens)
    print(f"[lda] {len(texts):,} documents loaded", file=sys.stderr)
    return doc_ids, texts


def train_lda(texts: list[list[str]], n_topics: int = 20, passes: int = 5):
    """
    Build gensim dictionary + corpus, train LDA model.

    Parameters
    ----------
    texts    : list of token lists (one per document)
    n_topics : number of latent topics to discover
    passes   : training passes (5 is enough for stable topics on large corpora)

    Returns
    -------
    lda_model, gensim_dictionary, bow_corpus (MmCorpus on disk, not a list)
    """
    try:
        from gensim import corpora
        from gensim.models import LdaModel
        from gensim.corpora import MmCorpus
    except ImportError:
        print("[lda] ERROR: gensim not installed. Run: pip install gensim", file=sys.stderr)
        sys.exit(1)

    print("[lda] building gensim dictionary …", file=sys.stderr)
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.4)
    print(f"[lda] dictionary after filtering: {len(dictionary):,} terms", file=sys.stderr)

    # Write bow corpus to a temp file on disk instead of holding it in RAM.
    # MmCorpus streams from disk during training — avoids holding 233k lists in memory.
    print("[lda] building bag-of-words corpus …", file=sys.stderr)
    bow_path = os.path.join(tempfile.gettempdir(), "threatsearch_bow.mm")
    MmCorpus.serialize(bow_path, (dictionary.doc2bow(t) for t in texts))
    bow_corpus = MmCorpus(bow_path)

    print(f"[lda] training LDA  n_topics={n_topics}  passes={passes} …", file=sys.stderr)
    t0 = time.perf_counter()
    lda_model = LdaModel(
        corpus       = bow_corpus,
        id2word      = dictionary,
        num_topics   = n_topics,
        passes       = passes,
        alpha        = "symmetric",   
        eta          = "auto",
        random_state = 42,
        chunksize    = 2000,          
    )
    elapsed = time.perf_counter() - t0
    print(f"[lda] training complete in {elapsed:.1f}s", file=sys.stderr)

    return lda_model, dictionary, bow_corpus


def compute_coherence(lda_model, texts, dictionary, sample: int = 20000) -> float:
    """
    Compute C_v coherence score on a sample of documents.
    Full-corpus coherence on 200k+ docs hangs — sampling gives a good
    enough estimate in seconds instead of hours.
    """
    try:
        import random
        from gensim.models import CoherenceModel
        sample_texts = random.sample(texts, min(sample, len(texts)))
        cm = CoherenceModel(
            model      = lda_model,
            texts      = sample_texts,
            dictionary = dictionary,
            coherence  = "c_v",
        )
        score = cm.get_coherence()
        print(f"[lda] coherence (C_v): {score:.4f}  (sampled {len(sample_texts):,} docs)",
              file=sys.stderr)
        return score
    except Exception as e:
        print(f"[lda] coherence computation failed: {e}", file=sys.stderr)
        return 0.0


def extract_topic_terms(lda_model, n_top_words: int = 15) -> list[dict]:
    """
    Return a list of topic dicts:
    [ { "topic_id": 0, "terms": [{"word": "...", "weight": 0.03}, ...] }, ... ]
    """
    topics = []
    for topic_id in range(lda_model.num_topics):
        top = lda_model.show_topic(topic_id, topn=n_top_words)
        topics.append({
            "topic_id": topic_id,
            "terms"   : [{"word": w, "weight": float(p)} for w, p in top],
        })
    return topics


def assign_doc_topics(lda_model, bow_corpus, doc_ids) -> list[dict]:
    """
    Assign each document to its dominant topic using batch inference.
    Uses get_document_topics on the full corpus at once — much faster
    than calling it once per document for large corpora.
    """
    assignments = []
    for doc_id, topic_dist in zip(doc_ids, lda_model[bow_corpus]):
        dominant = max(topic_dist, key=lambda x: x[1])
        assignments.append({
            "doc_id"         : doc_id,
            "dominant_topic" : int(dominant[0]),
        })
    return assignments


def save_outputs(
    lda_model,
    topics:      list[dict],
    assignments: list[dict],
    coherence:   float,
    out_dir:     str,
):
    os.makedirs(out_dir, exist_ok=True)

    lda_model.save(os.path.join(out_dir, "lda_model"))

    # Topic terms JSON
    topic_path = os.path.join(out_dir, "topic_terms.json")
    with open(topic_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2)

    # Doc→topic assignments JSON (no indent — 246k entries with indent=2 is ~80 MB)
    doc_path = os.path.join(out_dir, "doc_topics.json")
    with open(doc_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, separators=(",", ":"))

    # Coherence score
    coh_path = os.path.join(out_dir, "coherence.txt")
    with open(coh_path, "w") as f:
        f.write(f"C_v coherence: {coherence:.4f}\n")
        f.write(f"n_topics: {lda_model.num_topics}\n")
        f.write(f"n_docs: {len(assignments)}\n")

    print(f"[lda] outputs saved to {out_dir}/", file=sys.stderr)
    print(f"  topic_terms.json  — {len(topics)} topics", file=sys.stderr)
    print(f"  doc_topics.json   — {len(assignments):,} document assignments", file=sys.stderr)
    print(f"  coherence.txt     — C_v = {coherence:.4f}", file=sys.stderr)


def print_topics(topics: list[dict], n_show: int = 5):
    """Pretty-print top topics to stdout."""
    print(f"\n── Top {n_show} topics (by first term weight) ──────────────────")
    for t in topics[:n_show]:
        terms = ", ".join(w["word"] for w in t["terms"][:8])
        print(f"  Topic {t['topic_id']:02d}: {terms}")
    print()



def run(corpus_dir, out_dir, n_topics, passes, n_top_words, min_doc_len):
    doc_ids, texts = load_corpus_for_lda(corpus_dir, min_doc_len)

    lda_model, dictionary, bow_corpus = train_lda(texts, n_topics, passes)
    coherence  = compute_coherence(lda_model, texts, dictionary)
    topics     = extract_topic_terms(lda_model, n_top_words)
    assignments= assign_doc_topics(lda_model, bow_corpus, doc_ids)

    save_outputs(lda_model, topics, assignments, coherence, out_dir)
    print_topics(topics)
    return lda_model, topics, assignments, coherence


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ThreatSearch LDA topic modeler")
    ap.add_argument("--corpus-dir",   default="data/corpus")
    ap.add_argument("--out-dir",      default="data/lda")
    ap.add_argument("--topics",       default=20,  type=int, help="Number of LDA topics")
    ap.add_argument("--passes",       default=10,  type=int, help="Training passes")
    ap.add_argument("--top-words",    default=15,  type=int, help="Top words per topic to save")
    ap.add_argument("--min-doc-len",  default=5,   type=int, help="Skip docs shorter than this")
    args = ap.parse_args()

    run(
        corpus_dir  = args.corpus_dir,
        out_dir     = args.out_dir,
        n_topics    = args.topics,
        passes      = args.passes,
        n_top_words = args.top_words,
        min_doc_len = args.min_doc_len,
    )