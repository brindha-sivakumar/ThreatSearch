# ThreatSearch — Cybersecurity Advisory Retrieval System

---

## Overview

ThreatSearch is a domain-adapted information retrieval system that lets security analysts search and rank cybersecurity threat intelligence from two public sources: the National Vulnerability Database (NVD) and MITRE ATT&CK. Unlike a general-purpose search engine, ThreatSearch is built to understand security-specific language — it preserves CVE identifiers, ATT&CK technique IDs, and CWE weakness codes as atomic tokens, uses an expanded security stopword list, and expands queries using ATT&CK's own technique taxonomy.

The system extends a basic inverted index pipeline with a full retrieval stack: BM25 ranked retrieval, Boolean AND/OR/NOT query handling, ATT&CK-aware query expansion, LSA dimensionality reduction for memory-safe semantic search, and LDA topic modeling with threat-cluster visualization over the CVE corpus.

---

### Usage

```
python main.py --data-dir <path>   [--skip-ingest] [--skip-index]
               [--query <text>]    [--scorer {bm25,lsa}] [--no-expand]
```

| Argument | Purpose |
|---|---|
| `--data-dir` | Root directory containing `nvd/` and `attack/` (required) |
| `--skip-ingest` | Skip corpus ingestion — use existing `data/corpus/` shards |
| `--skip-index` | Skip index build — use existing `data/index/` files |
| `--query` | Run this search query and exit. Skips the pipeline build. |
| `--scorer` | `bm25` (default, exact keyword), `tfidf` (no length normalisation), or `lsa` (semantic similarity) |
| `--no-expand` | Disable ATT&CK query expansion — exact keyword search only |

---

**Starting from scratch (first run):**

Runs the full pipeline: ingestion → index → merge → LSA → LDA → visualization.
Requires `data/nvd/` and `data/attack/enterprise-attack.json` to be present.

```python
!python main.py --data-dir data
```

---

**Re-running with existing corpus (ingestion already done):**

Skips `ingest.py` and rebuilds the index, LSA, LDA, and visualizations.
Use this when you want to rebuild the index without re-downloading or re-parsing data.

```python
!python main.py --data-dir data --skip-ingest
```

---

**Re-running with existing index (index already built):**

Skips both ingestion and index build. Only re-runs LSA, LDA, and visualization.
Use this after the index is confirmed good and you only want to re-tune LDA or regenerate visualizations.

```python
!python main.py --data-dir data --skip-ingest --skip-index
```

---

**Querying — BM25 (default, exact keyword matching):**

Stems the query, looks up posting lists, scores using BM25. ATT&CK expansion injects
related technique IDs and CVE IDs as additional weighted terms. Returns top 10 results
with NVD and ATT&CK URLs.

```python
!python main.py --data-dir data --query "heap overflow remote code execution"
!python main.py --data-dir data --query "smb AND windows AND NOT authenticated"
!python main.py --data-dir data --query "T1190"
```

**Querying — TF-IDF:**

Simpler than BM25 — no length normalisation, so longer documents with more term
occurrences score higher. Useful for comparing against BM25 to see the effect of
length normalisation on result ordering.

```python
!python main.py --data-dir data --query "sql injection authentication" --scorer tfidf
```

**Querying — LSA (semantic similarity):**

Projects the query into a 300-dimensional latent space. Finds documents that are
semantically close even if they use different vocabulary. Requires the LSA index
to have been built (`data/lsa_index/` must exist). Returns cosine similarity scores
in the range [−1, 1] rather than BM25 weighted counts.

```python
!python main.py --data-dir data --query "memory corruption" --scorer lsa
!python main.py --data-dir data --query "credential theft" --scorer lsa
```

> To demonstrate LSA vs BM25, run the same query with both scorers and compare which
> CVEs surface. LSA will return documents using synonymous vocabulary (e.g. "use-after-free"
> for a "memory corruption" query) that BM25 misses entirely.

**Querying — without expansion:**

Disables ATT&CK query expansion. Use this when you want pure keyword retrieval or when
expansion is producing slow results on generic terms like "windows" or "remote".

```python
!python main.py --data-dir data --query "sql injection login" --no-expand
!python main.py --data-dir data --query "windows remote code execution" --no-expand
```

---

**Generating LDA visualizations (word clouds + tactic heatmap):**

LDA runs automatically as part of the full pipeline. To regenerate visualizations
after a pipeline run without rebuilding the index:

```python
# Run topic modeling and generate both visualizations
from topic_model import run as lda_run
lda_run(corpus_dir="data/corpus", out_dir="data/lda",
        n_topics=20, passes=10, n_top_words=15, min_doc_len=5)

# Generate visualizations from existing LDA outputs
from visualize import load_topic_terms, load_doc_topics, plot_wordclouds, plot_tactic_heatmap
import os
os.makedirs("data/viz", exist_ok=True)

topics     = load_topic_terms("data/lda")
doc_topics = load_doc_topics("data/lda")

# Word clouds — one per topic, word size proportional to LDA weight
plot_wordclouds(topics, "data/viz/topic_wordclouds.png")

# Tactic heatmap — ATT&CK kill-chain phases × LDA topics, cell = CVE count
plot_tactic_heatmap(
    "data/lda",
    "data/attack/enterprise-attack.json",
    "data/index/technique_cve_map.json",
    "data/viz/tactic_heatmap.png",
)
```

Display the outputs in Colab:

```python
from IPython.display import Image, display
display(Image("data/viz/topic_wordclouds.png", width=900))
display(Image("data/viz/tactic_heatmap.png",   width=900))
```

Inspect discovered topics directly:

```python
import json
with open("data/lda/topic_terms.json") as f:
    topics = json.load(f)
for t in topics:
    terms = ", ".join(w["word"] for w in t["terms"][:8])
    print(f"Topic {t['topic_id']:02d}: {terms}")

with open("data/lda/coherence.txt") as f:
    print(f"\nCoherence (C_v): {f.read().strip()}")
# C_v above 0.5 = interpretable topics; above 0.6 = good
```

---

Data directory layout expected
-------------------------------
    <data-dir>/
    ├── nvd/                ← NVD .json or .json.gz feeds  (required)
    └── attack/
        └── enterprise-attack.json   ← ATT&CK STIX bundle (required)

All outputs are written under <data-dir>/ automatically:
    <data-dir>/corpus/      ← ingested corpus shards
    <data-dir>/index/       ← inverted index files
    <data-dir>/lsa_index/   ← LSA matrices
    <data-dir>/lda/         ← LDA model and topic files
    <data-dir>/viz/         ← visualizations


### Setup

```python
pip install nltk scipy numpy gensim wordcloud matplotlib
import nltk
nltk.download('stopwords')
git clone https://github.com/brindha-sivakumar/ThreatSearch.git
```

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

**Outputs:** `data/corpus/*.txt` (corpus shards), `data/index/dictionary.txt`, `data/index/index_merged.txt`

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

```python
# 1c. Merge shards into a single global index (external k-way merge sort)
#     Each term appears exactly once in index_merged.txt
#     ranker.py picks this up automatically — no code changes needed
!python merge_index.py --index-dir data/index
```

**`ingest.py`**  
Reads NVD JSON/JSON.gz feeds and the ATT&CK STIX bundle. Extracts CVE descriptions, CWE tags, reference titles, technique names, detection notes, and platform data. Runs a pre-pass to extract structured identifiers (CVE-XXXX-XXXX, CWE-NNN, T1021.002) as atomic tokens before the regular lexer touches the text. Expands security acronyms (RCE → remotecodeexecution). Writes one corpus shard per 10,000 NVD documents plus one shard for ATT&CK.

**`nlp.py`**  
The security-aware NLP layer shared by all other components. Applies the same lexer/parser from hw1 extended with: structured ID detection (IDs bypass stemming and stopword filtering), an expanded security stopword list (words like "vulnerability", "advisory", "attacker" that appear in almost every document and add no discriminating value), Porter stemming on regular tokens, and a minimum token length filter. Every other component imports `process_line()` from this file.

**`index.py`**  
Two-pass inverted index builder. Pass 1 streams all corpus shards once to build the global vocabulary and write `dictionary.txt` — each word's line number is its word code, matching hw2's convention. Pass 2 re-streams each shard independently to build a local index file containing only terms with actual postings in that shard. Posting format: `<word-code> <word> <doc-frequency> (<doc-id>,<tf>,<source>) ...`.

**`merge_index.py`**  
External k-way merge sort over the per-shard index files. Each shard is already sorted by word code, so only the merge phase is needed. A min-heap holds one line per open shard file, producing `index_merged.txt` where every term appears exactly once with all postings combined. Memory is bounded to K buffered lines at any moment. `ranker.py` finds the merged file automatically via its `index_*.txt` glob pattern.

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

# With ATT&CK expansion disabled (faster, exact keyword only)
!python query.py "log4j jndi remote code execution" --no-expand

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

# LSA semantic search (requires lsa_build.py to have run first)
!python query.py "memory safety heap corruption" --scorer lsa --top 5
```

```python
# Verify the map has content
import json
with open("data/index/technique_cve_map.json") as f:
    m = json.load(f)
total = sum(len(v) for v in m.values())
print(f"{len(m)} techniques mapped to {total:,} CVEs")
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
Unified search interface. Auto-detects Boolean queries (from AND/OR/NOT keywords), strips operators before NLP processing, runs the Boolean handler if needed, then calls the expander and ranker. Supports BM25, TF-IDF, and LSA scorers via `--scorer`. Generates clickable NVD and ATT&CK URLs in the results output.

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
# Query using LSA via main.py
!python main.py --data-dir data/ --query "buffer overflow privilege escalation" --scorer lsa
!python main.py --data-dir data/ --query "lateral movement SMB" --scorer lsa --no-expand
```

**`lsa_build.py`**  
Streams all index shards into a `scipy.sparse` TF-IDF matrix (never a dense array). Applies sklearn's `TruncatedSVD` to compress to `n_components` latent dimensions. L2-normalises the resulting term and document matrices so cosine similarity reduces to a dot product at query time. Saves `term_matrix.npy` (V × k), `doc_matrix.npy` (N × k), `doc_ids.json`, and `words.json`.

**`lsa_ranker.py`**  
Loads the two numpy arrays at startup. Each query is a weighted average of the query terms' latent vectors, then a single matrix dot product against all document vectors produces cosine similarity scores. Memory usage is fixed at approximately `(V + N) × k × 4 bytes` regardless of query volume — for 300 components and the full corpus this is roughly 200 MB, and it never grows. This is the OOM management property described in the project proposal.

---

### Phase 3 — LDA topic modeling

**Prerequisite:** Phase 1 ingestion complete (`data/corpus/` must exist). Phase 2 and LSA are **not** required — `topic_model.py` reads corpus shards directly, not the inverted index.

**What this phase does:** Trains an LDA model over CVE descriptions to discover recurring vulnerability themes (e.g. memory corruption, authentication bypass, injection attacks) without any manual labelling. It uses its own `topic_tokenize()` preprocessing pipeline — deliberately separate from the index NLP — which skips stemming (so topic words stay readable), strips structured IDs like CVE-XXXX and CWE-NNN (which carry no topical meaning), and removes an extended list of CVE boilerplate words that survive standard stopword lists (`fix`, `call`, `make`, `prior`, `resolve`, etc.).

**Files involved:** `topic_model.py`

**Outputs:** `data/lda/lda_model`, `data/lda/topic_terms.json`, `data/lda/doc_topics.json`, `data/lda/coherence.txt`

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
# Check coherence score (written to data/lda/coherence.txt)
# C_v above 0.5 = interpretable topics; above 0.6 = good
with open("data/lda/coherence.txt") as f:
    print(f.read())
```

**`topic_model.py`**  
Trains an LDA model over CVE descriptions using gensim. Reads corpus shards from data/corpus/nvd_*.txt directly — the inverted index is not involved. Uses its own topic_tokenize() function that is deliberately separate from the index NLP pipeline: no stemming (so words stay readable as injection not inject), no structured IDs (CVE-XXXX and CWE-NNN are excluded as they carry no topical meaning), and an extended boilerplate stopword list targeting words common to almost every CVE description (fix, call, make, prior, resolve, etc.). Vocabulary is further filtered by filter_extremes — terms in fewer than 10 documents or more than 40% of documents are dropped. Writes topic_terms.json (top 15 words per topic with probability weights), doc_topics.json (dominant topic and confidence per CVE), and coherence.txt (C_v coherence score — above 0.5 indicates interpretable topics, above 0.6 is good). 

---

### Phase 3 — Visualization

**Prerequisite:** Phase 3 LDA complete (`data/lda/` must exist). The tactic heatmap additionally requires `data/index/technique_cve_map.json` (built by `expander.py build-map`).
**Outputs:** `data/viz/topic_wordclouds.png`, `data/viz/tactic_heatmap.png`

```python
# Install visualization dependencies
!pip install wordcloud matplotlib
```

```python
# Step 2: Generate visualizations
!python visualize.py

# Or generate individually:
!python visualize.py --only wordclouds   # word clouds only
!python visualize.py --only heatmap      # tactic heatmap only
# Note: pyLDAvis browser and CVE scatter plot are not included in this version
```
**What each visualization shows:**

| Output | What it shows | Best for |
|---|---|---|
| `topic_wordclouds.png` | One word cloud per topic; word size = LDA weight | Quick overview of all topics at once |
| `tactic_heatmap.png` | ATT&CK kill-chain tactics × LDA topics; cell = CVE count | Connecting vulnerability themes to attack stages |


**`visualize.py`**  
Two threat-cluster visualizations over the LDA outputs. (1) **Word clouds** — one per topic, word size proportional to LDA probability weight, rendered on a dark background for readability. (2) **Tactic heatmap** — matrix of ATT&CK kill-chain tactics (rows) × LDA topics (columns), cell value = CVE count; shows how vulnerability themes align with attack stages. The heatmap requires `technique_cve_map.json` and skips gracefully if it is absent. Topic labels use the format `T02: inject, overflow, heap` derived directly from the top LDA terms.
---

## Folder Structure After Full Run

```
Project/
├── main.py
├── ingest.py          nlp.py         index.py
├── merge_index.py     run_doc_lengths.py
├── expander.py        ranker.py      boolean_query.py
├── query.py           lsa_build.py   lsa_ranker.py
├── topic_model.py     visualize.py
└── data/
    ├── nvd/                    ← NVD .json.gz feed files (manual download)
    ├── attack/
    │   └── enterprise-attack.json
    ├── corpus/                 ← created by ingest.py
    │   ├── nvd_0000.txt
    │   └── attack_0000.txt
    ├── index/                  ← created by index.py + merge_index.py
    │   ├── dictionary.txt
    │   ├── index_merged.txt
    │   ├── doc_lengths.json
    │   └── technique_cve_map.json
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
    ├── viz/                    ← created by visualize.py
    │   ├── topic_wordclouds.png
    │   └── tactic_heatmap.png
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
shutil.copytree('data/lda',       '/content/drive/MyDrive/threatsearch/lda')

# On next session, restore from Drive
shutil.copytree('/content/drive/MyDrive/threatsearch/index',     'data/index')
shutil.copytree('/content/drive/MyDrive/threatsearch/lsa_index', 'data/lsa_index')
shutil.copytree('/content/drive/MyDrive/threatsearch/lda',       'data/lda')
```

---