---

# ThreatSearch: Security-Aware Information Retrieval

A specialized search engine designed to ingest, index, and rank security-related data from the NVD (CVEs) and MITRE ATT&CK frameworks.

---

## Phase 1 — Corpus Ingestion & Inverted Index

Focuses on processing large-scale security datasets into a specialized inverted index.

### 1. Data Ingestion (`ingest.py`)
Streams NVD CVE feeds and MITRE ATT&CK STIX into a unified document format.

* **Setup:**
    * **NVD CVE feeds:** Download annual JSON.gz files from [NVD Data Feeds](https://nvd.nist.gov/vuln/data-feeds) and save to `data/nvd/`.
    * **MITRE ATT&CK:** Download the [STIX bundle](https://github.com/mitre/cti/blob/master/enterprise-attack/enterprise-attack.json) and save to `data/attack/enterprise-attack.json`.
* **Usage:**
    ```bash
    python ingest.py --nvd-dir data/nvd/ --attack data/attack/enterprise-attack.json --out-dir data/corpus/
    ```
* **Output:** One text file per source shard. Format: `<doc-id> <token1> <token2> ...`

### 2. Inverted Indexing (`index.py`)
A two-pass builder that creates a global vocabulary and source-sharded index files.

* **Pass 1:** Builds a global `dictionary.txt` (stemmed, alphabetical).
* **Pass 2:** Generates index lines: `<word-code> <word> <df> (<doc-id>, <tf>, <source>) ...`
* **Usage:**
    ```bash
    python index.py --corpus-dir data/corpus/ --out-dir data/index/
    ```

### 3. NLP Layer (`nlp.py`)
The security-aware pipeline that sits between ingestion and indexing.
* **Atomic IDs:** Preserves CVE/CWE/ATT&CK IDs without splitting.
* **Security Stopwords:** Filters common security domain noise.

---

## Phase 2 — Ranked Retrieval & Query Expansion

### 1. Document Lengths (`run_doc_lengths.py`)
Computes per-document token counts required for BM25 normalization. Run this once after Phase 1.
```bash
python run_doc_lengths.py --corpus-dir data/corpus/ --out data/index/doc_lengths.json
```

### 2. Ranking Engine (`ranker.py`)
Ranks documents using **BM25** (default) or **TF-IDF**.
* **Lazy Loading:** Index shards are loaded per query term to save memory.
* **Source Weighting:** ATT&CK postings receive a default **1.3× boost** for higher signal.

### 3. Query Expansion (`expander.py`)
Broadens search recall by mapping query terms to related MITRE Techniques and CVEs.
* **Technique Weights:** Matched keywords inject Technique IDs (Weight: 0.6).
* **CVE Weights:** Technique IDs inject associated CVE IDs (Weight: 0.4).

### 4. Search Interface (`query.py`)
The primary CLI for performing searches.
```bash
# Basic search
python query.py "buffer overflow SMB"

# Search with explanation and specific scorer
python query.py "T1059 powershell" --explain --scorer tfidf
```

---

## Phase 3 — Latent Semantic Analysis (LSA)

The LSA layer provides semantic search capabilities (finding synonyms/paraphrases) while maintaining a fixed memory footprint.

### 1. LSA Model Builder (`lsa_build.py`)
Compresses the inverted index into a latent-semantic index.
* **Strategy:** Applies `TruncatedSVD` to a sparse TF-IDF matrix.
* **Memory Efficiency:** Generates two fixed-size numpy arrays (~200 MB total for 300 components).
* **Usage:**
    ```bash
    python lsa_build.py --index-dir data/index --out-dir data/lsa_index --components 300
    ```

### 2. LSA Query Engine (`lsa_ranker.py`)
Uses cosine similarity in latent space to find documents semantically related to the query.
* **OOM Management:** Unlike the BM25 ranker, memory usage remains constant regardless of query volume.
* **Usage:**
    ```bash
    python query.py "buffer overflow SMB" --scorer lsa
    ```

---

## Project Structure
```text
data/
├── nvd/            # Raw NVD JSON feeds
├── attack/         # MITRE STIX JSON
├── corpus/         # Processed shards (ingest.py)
├── index/          # Inverted index & dictionary (index.py)
└── lsa_index/      # Compressed LSA matrices (lsa_build.py)
```