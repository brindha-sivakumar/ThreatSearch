import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path


# ── Security-specific token pre-processing ────────────────────────────────────

# These patterns are extracted BEFORE the regular lexer runs so that
# structured identifiers are never split or stripped of digits.
_ID_PATTERNS = [
    re.compile(r'\bCVE-\d{4}-\d+\b', re.IGNORECASE),       # CVE-2021-44228
    re.compile(r'\bCWE-\d+\b', re.IGNORECASE),              # CWE-79
    re.compile(r'\bCAPEC-\d+\b', re.IGNORECASE),            # CAPEC-86
    re.compile(r'\bT\d{4}(?:\.\d{3})?\b'),                  # T1021 / T1021.002
    re.compile(r'\bTA\d{4}\b'),                              # TA0001
    re.compile(r'\bMS\d{2}-\d+\b', re.IGNORECASE),          # MS17-010
]

# Security acronyms expanded to their full form so the stemmer handles them
# consistently and they survive the alpha-only filter.
_ACRONYMS = {
    "rce":   "remotecodeexecution",
    "lpe":   "localprivilegeescalation",
    "sqli":  "sqlinjection",
    "xss":   "crosssitescripting",
    "csrf":  "crosssiterequestforgery",
    "ssrf":  "serversiderequestforgery",
    "lfi":   "localfileinclusion",
    "rfi":   "remotefileinclusion",
    "dos":   "denialofservice",
    "ddos":  "distributeddenialofservice",
    "mitm":  "maninthemiddle",
    "ttp":   "tacticstechniquesprocedures",
    "ioc":   "indicatorofcompromise",
    "c2":    "commandcontrol",
    "poc":   "proofofconcept",
    "cve":   "commonvulnerabilityexposure",
    "cwe":   "commonweaknessenumeration",
    "cvss":  "commonvulnerabilityscore",
}


def extract_security_ids(text: str) -> list[str]:
    """Return all structured security identifiers found in text as atomic tokens."""
    ids = []
    for pattern in _ID_PATTERNS:
        ids.extend(m.group(0).upper() for m in pattern.finditer(text))
    return ids


def expand_acronyms(token: str) -> str:
    """Replace known security acronyms with their expanded form."""
    return _ACRONYMS.get(token.lower(), token)


# ── NVD ingestion ─────────────────────────────────────────────────────────────

def _nvd_text(item: dict) -> str:
    """Extract all useful text fields from one NVD CVE item."""
    parts = []

    # English description
    for desc in item.get("cve", {}).get("description", {}).get("description_data", []):
        if desc.get("lang") == "en":
            parts.append(desc.get("value", ""))

    # CWE names (weakness type, very informative)
    for pd in item.get("cve", {}).get("problemtype", {}).get("problemtype_data", []):
        for d in pd.get("description", []):
            if d.get("lang") == "en":
                parts.append(d.get("value", ""))

    # Reference titles / tags (e.g. "Exploit", "Patch", "Vendor Advisory")
    for ref in item.get("cve", {}).get("references", {}).get("reference_data", []):
        name = ref.get("name", "")
        if name and not name.startswith("http"):
            parts.append(name)
        for tag in ref.get("tags", []):
            parts.append(tag)

    return " ".join(parts)


def stream_nvd(nvd_dir: str):
    """
    Yield (doc_id, raw_text) for every CVE in all .json.gz files under nvd_dir.
    Handles both NVD API 1.1 feed format (cve/CVE_Items) and 2.0 (vulnerabilities).
    """
    nvd_path = Path(nvd_dir)
    files = sorted(nvd_path.glob("*.json.gz")) + sorted(nvd_path.glob("*.json"))
    if not files:
        print(f"[ingest] WARNING: no NVD files found in {nvd_dir}", file=sys.stderr)
        return

    for fpath in files:
        print(f"[ingest] NVD: {fpath.name}", file=sys.stderr)
        opener = gzip.open if fpath.suffix == ".gz" else open
        with opener(fpath, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        # NVD 1.1 format
        items = data.get("CVE_Items", [])
        for item in items:
            cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID", "")
            if not cve_id:
                continue
            yield cve_id.upper(), _nvd_text(item)

        # NVD 2.0 format
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id = cve.get("id", "")
            if not cve_id:
                continue
            desc = " ".join(
                d["value"] for d in cve.get("descriptions", [])
                if d.get("lang") == "en"
            )
            weaknesses = " ".join(
                d["description"][0]["value"]
                for d in cve.get("weaknesses", [])
                if d.get("description")
            )
            yield cve_id.upper(), f"{desc} {weaknesses}"


# ── ATT&CK ingestion ──────────────────────────────────────────────────────────

def _attack_text(obj: dict) -> str:
    """Extract indexable text from one ATT&CK STIX object."""
    parts = []
    parts.append(obj.get("name", ""))

    for ref in obj.get("external_references", []):
        parts.append(ref.get("description", ""))

    for d in obj.get("description", "").split("\n"):
        parts.append(d)

    # Detection and mitigation fields (rich signal for security queries)
    parts.append(obj.get("x_mitre_detection", ""))
    for p in obj.get("x_mitre_platforms", []):
        parts.append(p)
    for d in obj.get("x_mitre_data_sources", []):
        parts.append(d)
    for p in obj.get("kill_chain_phases", []):
        parts.append(p.get("phase_name", ""))

    return " ".join(parts)


def _attack_id(obj: dict) -> str | None:
    """Return the ATT&CK external ID (e.g. T1021.002) or None."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "").upper()
    return None


def stream_attack(attack_path: str):
    """
    Yield (doc_id, raw_text) for every technique and sub-technique in the
    ATT&CK STIX bundle.  Skips revoked or deprecated entries.
    """
    with open(attack_path, encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle if isinstance(bundle, list) else bundle.get("objects", [])
    print(f"[ingest] ATT&CK: {len(objects)} STIX objects", file=sys.stderr)

    for obj in objects:
        if obj.get("type") not in ("attack-pattern",):
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = _attack_id(obj)
        if not tid:
            continue
        yield tid, _attack_text(obj)


def tokenize_raw(doc_id: str, raw_text: str) -> str:
    """
    Convert (doc_id, raw_text) into a corpus line:
        <doc-id> <token1> <token2> ...

    Security IDs are extracted first as atomic tokens, then the remaining
    text is split into whitespace-delimited tokens for the downstream NLP
    pipeline (nlp.py) to clean and stem.
    """
    # 1. Pull structured IDs out first (they must survive intact)
    security_ids = extract_security_ids(raw_text)

    # 2. Remaining text: strip the IDs to avoid double-counting, then split
    remaining = raw_text
    for pat in _ID_PATTERNS:
        remaining = pat.sub(" ", remaining)

    word_tokens = remaining.split()

    # 3. Expand known acronyms
    word_tokens = [expand_acronyms(t) for t in word_tokens]

    all_tokens = security_ids + word_tokens
    return doc_id + " " + " ".join(all_tokens)


def write_shard(lines: list[str], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"[ingest] wrote {len(lines):,} docs → {out_path}", file=sys.stderr)


def ingest(nvd_dir: str | None, attack_path: str | None, out_dir: str,
           shard_size: int = 10_000):
    """
    Ingest NVD and/or ATT&CK, write corpus shards to out_dir.
    Returns list of output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_files = []

    if nvd_dir:
        buffer, shard_idx = [], 0
        for doc_id, raw_text in stream_nvd(nvd_dir):
            buffer.append(tokenize_raw(doc_id, raw_text))
            if len(buffer) >= shard_size:
                path = os.path.join(out_dir, f"nvd_{shard_idx:04d}.txt")
                write_shard(buffer, path)
                out_files.append(path)
                buffer, shard_idx = [], shard_idx + 1
        if buffer:
            path = os.path.join(out_dir, f"nvd_{shard_idx:04d}.txt")
            write_shard(buffer, path)
            out_files.append(path)

    if attack_path:
        buffer = []
        for doc_id, raw_text in stream_attack(attack_path):
            buffer.append(tokenize_raw(doc_id, raw_text))
        # ATT&CK is small (~700 techniques) — one shard
        path = os.path.join(out_dir, "attack_0000.txt")
        write_shard(buffer, path)
        out_files.append(path)

    return out_files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ThreatSearch ingestion")
    ap.add_argument("--nvd-dir",  help="Directory of NVD JSON/JSON.gz feed files")
    ap.add_argument("--attack",   help="Path to ATT&CK STIX bundle JSON")
    ap.add_argument("--out-dir",  default="data/corpus", help="Output directory for corpus shards")
    ap.add_argument("--shard-size", type=int, default=10_000, help="Docs per NVD shard")
    args = ap.parse_args()

    if not args.nvd_dir and not args.attack:
        ap.error("Provide at least one of --nvd-dir or --attack")

    files = ingest(args.nvd_dir, args.attack, args.out_dir, args.shard_size)
    print(f"\n[ingest] complete — {len(files)} shard file(s) written to {args.out_dir}")
