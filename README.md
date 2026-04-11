# ThreatSearch — Cybersecurity Advisory Retrieval System

---

## Overview

ThreatSearch is a domain-adapted information retrieval system that lets security analysts search and rank cybersecurity threat intelligence from two public sources: the National Vulnerability Database (NVD) and MITRE ATT&CK. Unlike a general-purpose search engine, ThreatSearch is built to understand security-specific language — it preserves CVE identifiers, ATT&CK technique IDs, and CWE weakness codes as atomic tokens, uses an expanded security stopword list, and expands queries using ATT&CK's own technique taxonomy.

The system extends a basic inverted index pipeline with a full retrieval stack: BM25 ranked retrieval, Boolean AND/OR/NOT query handling, ATT&CK-aware query expansion, LSA dimensionality reduction for memory-safe semantic search, and LDA topic modeling over the CVE corpus.

---
### Setup

pip install nltk scikit-learn scipy numpy gensim
import nltk
nltk.download('stopwords')
git clone https://github.com/brindha-sivakumar/ThreatSearch.git

### Download the datasets

#### NVD CVE feeds

```python
import os, requests

os.makedirs("data/nvd", exist_ok=True)

# Download one year as a test (add more years as needed)
years = [2021, 2022, 2023, 2024]   # adjust to your needs

for year in years:
    url  = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
    dest = f"data/nvd/nvdcve-1.1-{year}.json.gz"
    print(f"Downloading {year}...")
    r = requests.get(url, stream=True, timeout=120)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  saved {os.path.getsize(dest) // 1024:,} KB")

print("NVD download complete.")
```

> **Tip:** Start with just 2021 to test the pipeline before downloading all years. Each file is 40–200 MB.

#### MITRE ATT&CK

```python
import requests, os

os.makedirs("data/attack", exist_ok=True)

url  = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
dest = "data/attack/enterprise-attack.json"

print("Downloading ATT&CK bundle (~75 MB)...")
r = requests.get(url, stream=True, timeout=300)
with open(dest, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
print(f"Saved {os.path.getsize(dest) // 1024:,} KB → {dest}")
```

---

## Running the Pipeline

---

### Phase 1 — Corpus ingestion and inverted index

**What this phase does:** Reads raw NVD JSON feeds and the ATT&CK STIX bundle, applies security-aware NLP (ID preservation, stemming, stopword filtering, acronym expansion), and builds a sharded inverted index plus a global dictionary.

**Files involved:** `ingest.py` → `nlp.py` → `index.py`

**Outputs:** `data/corpus/*.txt` (corpus shards), `data/index/dictionary.txt`, `data/index/index_*.txt`

```python
# 1a. Ingest: convert raw data to corpus shards
!python ingest.py \
    --nvd-dir data/nvd/ \
    --attack  data/attack/enterprise-attack.json \
    --out-dir data/corpus/
```

```python
# 1b. Build index: two-pass over corpus shards
#     Pass 1 → dictionary.txt (global vocabulary, word codes)
#     Pass 2 → index_*.txt   (per-shard posting lists)
!python index.py \
    --corpus-dir data/corpus/ \
    --out-dir    data/index/
```

**`ingest.py`**  
Reads NVD JSON/JSON.gz feeds and the ATT&CK STIX bundle. Extracts CVE descriptions, CWE tags, reference titles, technique names, detection notes, and platform data. Runs a pre-pass to extract structured identifiers (CVE-XXXX-XXXX, CWE-NNN, T1021.002) as atomic tokens before the regular lexer touches the text. Expands security acronyms (RCE → remotecodeexecution). Writes one corpus shard per 10,000 NVD documents plus one shard for ATT&CK.

**`nlp.py`**  
The security-aware NLP layer shared by all other components. Applies the same lexer/parser from hw1 extended with: structured ID detection (IDs bypass stemming and stopword filtering), an expanded security stopword list (words like "vulnerability", "advisory", "attacker" that appear in almost every document and add no discriminating value), Porter stemming on regular tokens, and a minimum token length filter. Every other component imports `process_line()` from this file.

**`index.py`**  
Two-pass inverted index builder. Pass 1 streams all corpus shards once to build the global vocabulary and write `dictionary.txt` — each word's line number is its word code, matching hw2's convention. Pass 2 re-streams each shard independently to build a local index file containing only terms with actual postings in that shard. Posting format: `<word-code> <word> <doc-frequency> (<doc-id>,<tf>,<source>) ...`. Includes a `verify_word_codes()` function that cross-checks every shard against the dictionary after writing.

---
### Phase 2 — Ranked retrieval and query expansion

**What this phase does:** Adds BM25 scoring over the posting lists, ATT&CK-aware query expansion (maps query terms to ATT&CK technique IDs and injects related CVEs), and Boolean AND/OR/NOT query handling. `query.py` is the unified interface for all query modes.

**Files involved:** `run_doc_lengths.py`, `expander.py`, `ranker.py`, `boolean_query.py`, `query.py`

**Outputs:** `data/index/doc_lengths.json`, `data/index/technique_cve_map.json`

```python
# 2a. Pre-compute document lengths (needed by BM25 length normalisation)
!python run_doc_lengths.py
```

```python
# 2b. Build technique→CVE map (enables CVE injection during expansion)
!python expander.py build-map
```

```python
# 2c. Run example queries

# Standard BM25 query
!python query.py "buffer overflow SMB"

# With ATT&CK expansion details
!python query.py "log4j jndi remote code execution" --explain

# Boolean AND — both terms must appear
!python query.py "smb AND windows"

# Boolean OR — either term matches
!python query.py "SMB OR RDP"

# Boolean AND NOT — exclude term
!python query.py "windows AND NOT authenticated"

# Boolean with parentheses
!python query.py "windows AND (smb OR rdp)"

# TF-IDF instead of BM25, no expansion
!python query.py "lateral movement credential" --scorer tfidf --no-expand --top 20
```
**`run_doc_lengths.py`**  
Scans all corpus shards and computes the token count per document. Writes `doc_lengths.json`. BM25's length normalisation term requires knowing each document's length relative to the corpus average — without this file the BM25 scorer falls back to the corpus average for every document.

**`ranker.py`**  
BM25 and TF-IDF scorer. Loads posting lists lazily per query term from the index shard files and caches them in an LRU cache (capped at 2000 terms) to prevent unbounded memory growth. ATT&CK postings receive a 1.3× source weight boost since technique descriptions are higher-signal than generic CVE text. Accepts an optional `candidate_ids` set from the Boolean handler to restrict scoring to a pre-filtered document set.

**`boolean_query.py`**  
Boolean AND / OR / NOT query handler. Contains a lexer, a recursive descent parser that produces an AST, and an evaluator that applies set operations over posting sets. Supports parenthesised sub-expressions and `AND NOT` as a single operator. Returns a set of matching doc IDs that `ranker.py` uses as a candidate filter for BM25 re-ranking.

**`expander.py`**  
ATT&CK query expander. At startup, indexes every ATT&CK technique by the stemmed keywords in its name, description, detection text, and platform list. At query time, maps each query term to matching technique IDs (injected at 0.6× weight) and optionally to CVE IDs that co-occur with those techniques (injected at 0.4× weight). The `build-map` subcommand scans the corpus to build the technique→CVE mapping.

**`query.py`**  
Unified search interface. Auto-detects Boolean queries (from AND/OR/NOT keywords), strips operators before NLP processing, runs the Boolean handler if needed, then calls the expander and ranker. Supports single-query mode and interactive mode (loads the index once, much faster for multiple queries). Generates clickable NVD and ATT&CK URLs in the results output.
---

### LSA — Memory-safe semantic index

**What this phase does:** Builds a compressed latent-semantic representation of the entire corpus. Instead of growing a posting-list cache in RAM, the LSA ranker loads two fixed numpy arrays at startup and never allocates additional memory regardless of query volume. This is the OOM management approach described in the project proposal.

BM25 and LSA are complementary: BM25 is exact and fast; LSA handles synonyms and paraphrases that exact keyword matching misses. `query.py` supports both via `--scorer`.

**Files involved:** `lsa_build.py`, `lsa_ranker.py`

**Outputs:** `data/lsa_index/term_matrix.npy`, `data/lsa_index/doc_matrix.npy`, `data/lsa_index/doc_ids.json`, `data/lsa_index/words.json`

```python
# Build LSA index (5–20 min on full corpus)
# Use --components 100 for a faster test run
!python lsa_build.py --components 300
```

```python
# Query using LSA
!python lsa_ranker.py --lsa-dir data/lsa_index "buffer overflow privilege escalation"
!python lsa_ranker.py --lsa-dir data/lsa_index --explain "lateral movement SMB"
```

```python
# Memory comparison: run many queries and observe memory stays fixed
from lsa_ranker import LSARanker
ranker = LSARanker("data/lsa_index", "data/index/dictionary.txt")

mem = ranker.memory_usage_mb()
print(f"Memory before queries: {mem:.1f} MB")

queries = ["smb exploit", "log4j injection", "powershell bypass", "rdp bluekeep"]
for q in queries:
    from nlp import process_line
    _, terms = process_line("Q " + q)
    ranker.query(terms, top_n=5)

print(f"Memory after  queries: {ranker.memory_usage_mb():.1f} MB  (unchanged = OOM safe)")
```

**`lsa_build.py`**  
Streams all index shards into a `scipy.sparse` TF-IDF matrix (never a dense array). Applies sklearn's `TruncatedSVD` to compress to `n_components` latent dimensions. L2-normalises the resulting term and document matrices so cosine similarity reduces to a dot product at query time. Saves `term_matrix.npy` (V × k), `doc_matrix.npy` (N × k), `doc_ids.json`, and `words.json`.

**`lsa_ranker.py`**  
Loads the two numpy arrays at startup. Each query is a weighted average of the query terms' latent vectors, then a single matrix dot product against all document vectors produces cosine similarity scores. Memory usage is fixed at approximately `(V + N) × k × 4 bytes` regardless of query volume — for 300 components and the full corpus this is roughly 200 MB, and it never grows. This is the OOM management property described in the project proposal.

---

### Phase 3 — Analysis and evaluation

**What this phase does:** LDA topic modeling discovers recurring vulnerability themes in the CVE corpus without supervision. The evaluation framework issues ATT&CK technique names as queries and measures retrieval quality (Precision@K, MRR, nDCG@K) for both BM25 and LSA, producing a comparison table for the final report.

**Files involved:** `topic_model.py`, `evaluate.py`

**Outputs:** `data/lda/topic_terms.json`, `data/lda/doc_topics.json`, `data/eval/eval_summary.csv`

```python
# Install gensim (needed for LDA only)
!pip install gensim
```

```python
# LDA topic modeling over CVE corpus
# --topics: number of latent topics to discover
# --passes: training iterations (more = better coherence score, slower)
!python topic_model.py --topics 20 --passes 10
```

```python
# Inspect discovered topics
import json
with open("data/lda/topic_terms.json") as f:
    topics = json.load(f)

for t in topics[:10]:
    terms = ", ".join(w["word"] for w in t["terms"][:8])
    print(f"Topic {t['topic_id']:02d}: {terms}")
```

```python
# Assign a query to its closest topic
!python topic_model.py --query "heap overflow privilege escalation kernel"
```

```python
# Evaluation: BM25 vs LSA using ATT&CK ground truth
# Outputs a comparison table to stdout and eval_summary.csv
!python evaluate.py --scorer both --top-k 10
```


**`topic_model.py`**  
Trains an LDA model over CVE descriptions using gensim. Filters out very short documents and applies frequency-based vocabulary filtering (terms in fewer than 5 documents or more than 50% of documents are dropped as noise). Writes `topic_terms.json` (top 15 words per topic with weights), `doc_topics.json` (dominant topic and confidence per document), and `coherence.txt` (C_v coherence score — above 0.5 indicates interpretable topics). Also supports `--query` mode to assign a new text to its closest topic.

**`evaluate.py`**  
Evaluation framework. Uses the technique→CVE map from `expander.py` as ground truth: each ATT&CK technique name is issued as a query, and its mapped CVEs are the relevant documents. Computes Precision@K (fraction of top-K results that are relevant), MRR (reciprocal rank of the first relevant result), and nDCG@K (position-weighted relevance score) for both BM25 and LSA. Writes `eval_results.json` (per-query breakdown) and `eval_summary.csv` (aggregate comparison table for the report).

---

## Folder Structure After Full Run

```
Project/
├── ingest.py          nlp.py         index.py
├── run_doc_lengths.py expander.py    ranker.py
├── boolean_query.py   query.py
├── lsa_build.py       lsa_ranker.py
├── topic_model.py     evaluate.py
├── test_phase1.py     test_phase2.py
├── test_lsa.py        test_phase3.py     test_boolean.py
└── data/
    ├── nvd/                    ← NVD .json.gz feed files (manual download)
    ├── attack/
    │   └── enterprise-attack.json
    ├── corpus/                 ← created by ingest.py
    │   ├── nvd_0000.txt
    │   └── attack_0000.txt
    ├── index/                  ← created by index.py
    │   ├── dictionary.txt
    │   ├── doc_lengths.json
    │   ├── technique_cve_map.json
    │   └── index_nvd_*.txt  index_attack_*.txt
    ├── lsa_index/              ← created by lsa_build.py
    │   ├── term_matrix.npy
    │   ├── doc_matrix.npy
    │   ├── doc_ids.json
    │   └── words.json
    ├── lda/                    ← created by topic_model.py
    │   ├── lda_model
    │   ├── topic_terms.json
    │   ├── doc_topics.json
    │   └── coherence.txt
    └── eval/                   ← created by evaluate.py
        ├── eval_results.json
        └── eval_summary.csv
```

---

## Persisting Data Across Colab Sessions (Optional)

Colab sessions reset periodically. To avoid re-downloading and re-indexing:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy built index to Drive after Phase 1
import shutil
shutil.copytree('data/index',     '/content/drive/MyDrive/threatsearch/index')
shutil.copytree('data/lsa_index', '/content/drive/MyDrive/threatsearch/lsa_index')

# On next session, restore from Drive
shutil.copytree('/content/drive/MyDrive/threatsearch/index',     'data/index')
shutil.copytree('/content/drive/MyDrive/threatsearch/lsa_index', 'data/lsa_index')
```

---

