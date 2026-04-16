import json
import os
import re
import sys
from collections import defaultdict

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

_ps        = PorterStemmer()
_STOPS     = set(stopwords.words("english"))

TECHNIQUE_WEIGHT = 0.6
CVE_WEIGHT       = 0.4

# Minimum keyword length to add to the lookup table (avoids noise from short words)
_MIN_KW_LEN = 4


def _stem(word: str) -> str:
    return _ps.stem(word.lower())


def _keywords_from_text(text: str) -> set[str]:
    """Extract stemmed, stop-filtered keywords from a block of text."""
    tokens = re.findall(r'[a-zA-Z]{4,}', text)
    return {_stem(t) for t in tokens if t.lower() not in _STOPS}


def _attack_external_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "").upper()
    return None


class QueryExpander:
    """
    Builds ATT&CK keyword lookup at init, then expands queries at search time.

    Parameters
    ----------
    attack_path : path to enterprise-attack.json (already downloaded in Phase 1)
    nvd_mapping : optional path to a JSON file {technique_id: [cve_id, ...]}
                  If not provided, CVE injection is skipped.
    """

    def __init__(self, attack_path: str, nvd_mapping: str | None = None):
        # keyword (stemmed) → {technique_id, ...}
        self._kw_to_techniques: dict[str, set[str]] = defaultdict(set)

        # technique_id → {cve_id, ...}
        self._technique_to_cves: dict[str, set[str]] = defaultdict(set)

        # technique_id → human name (for display)
        self._technique_names: dict[str, str] = {}

        print("[expander] building ATT&CK keyword index …", file=sys.stderr)
        self._build_from_attack(attack_path)

        if nvd_mapping and os.path.exists(nvd_mapping):
            print("[expander] loading technique→CVE mapping …", file=sys.stderr)
            self._load_nvd_mapping(nvd_mapping)

        total_kw = len(self._kw_to_techniques)
        total_t  = len(self._technique_names)
        print(f"[expander] {total_t} techniques, {total_kw:,} keywords", file=sys.stderr)

    def _build_from_attack(self, attack_path: str):
        with open(attack_path, encoding="utf-8") as f:
            bundle = json.load(f)

        objects = bundle if isinstance(bundle, list) else bundle.get("objects", [])

        for obj in objects:
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue

            tid = _attack_external_id(obj)
            if not tid:
                continue

            name = obj.get("name", "")
            self._technique_names[tid] = name

            # Collect indexable text: name + description + detection + platforms
            text_parts = [
                name,
                obj.get("description", ""),
                obj.get("x_mitre_detection", ""),
                " ".join(obj.get("x_mitre_platforms", [])),
                " ".join(obj.get("x_mitre_data_sources", [])),
            ]
            full_text = " ".join(text_parts)
            keywords  = _keywords_from_text(full_text)

            for kw in keywords:
                if len(kw) >= _MIN_KW_LEN:
                    self._kw_to_techniques[kw].add(tid)

            # Also index the technique ID itself so users can search "T1021"
            self._kw_to_techniques[tid.upper()].add(tid)
            # And the stem of the technique name words
            for word in name.lower().split():
                stemmed = _stem(word)
                if len(stemmed) >= _MIN_KW_LEN and stemmed not in _STOPS:
                    self._kw_to_techniques[stemmed].add(tid)

    def _load_nvd_mapping(self, path: str):
        """
        Load a pre-built technique→CVE mapping.
        Format: {"T1021.002": ["CVE-2017-0144", ...], ...}
        See build_technique_cve_map.py for how to generate this.
        """
        with open(path, encoding="utf-8") as f:
            mapping = json.load(f)
        for tid, cves in mapping.items():
            self._technique_to_cves[tid.upper()].update(cves)
        total_cves = sum(len(v) for v in self._technique_to_cves.values())
        print(
            f"[expander] {len(self._technique_to_cves)} techniques "
            f"mapped to {total_cves:,} CVEs",
            file=sys.stderr,
        )

    def expand(
        self,
        query_terms: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        """
        Expand a list of processed query terms.

        Returns
        -------
        expanded_terms  : original terms + injected technique/CVE IDs
        term_weights    : {term: weight} — original terms have weight 1.0,
                          injected technique IDs have TECHNIQUE_WEIGHT,
                          injected CVE IDs have CVE_WEIGHT
        """
        weights: dict[str, float] = {t: 1.0 for t in query_terms}
        matched_techniques: set[str] = set()

        for term in query_terms:
            # Direct lookup (term already stemmed by nlp.py)
            hits = self._kw_to_techniques.get(term, set())
            matched_techniques.update(hits)

        # Inject technique IDs
        for tid in matched_techniques:
            if tid not in weights:
                weights[tid] = TECHNIQUE_WEIGHT
            else:
                weights[tid] = max(weights[tid], TECHNIQUE_WEIGHT)

            # Inject mapped CVE IDs
            for cve_id in self._technique_to_cves.get(tid, set()):
                if cve_id not in weights:
                    weights[cve_id] = CVE_WEIGHT

        expanded_terms = list(weights.keys())
        return expanded_terms, weights

    
# ── Build technique→CVE map ───────────────────────────────────────────────────


def build_technique_cve_map(
    attack_path: str,
    out_path:    str,
    corpus_dir:  str | None = None,
    min_overlap: int = 3,
):
    """
    Build technique→CVE relevance map via keyword overlap

       For each NVD CVE, counts how many stemmed tokens it shares with each
    ATT&CK technique's name + description. If the overlap meets min_overlap,
    the CVE is recorded as relevant to that technique.
 

    Parameters
    ----------
    attack_path : path to enterprise-attack.json
    out_path    : where to write technique_cve_map.json
    corpus_dir  : path to corpus shards (enables keyword-overlap fallback)
    min_overlap : minimum shared keyword count for keyword-based matching
    """
    import glob
    from nlp import process_line as _pl

    with open(attack_path, encoding="utf-8") as f:
        bundle = json.load(f)
    objects = bundle if isinstance(bundle, list) else bundle.get("objects", [])

    cve_pat = re.compile(r'^CVE-\d{4}-\d+$', re.IGNORECASE)
    
    
    # Build keyword set per technique from name + description + detection + platforms
    technique_keywords: dict[str, set[str]] = {}
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = _attack_external_id(obj)
        if not tid:
            continue
        text = " ".join([
            obj.get("name", ""),
            obj.get("description", "")[:500],
            obj.get("x_mitre_detection", "")[:200],
            " ".join(obj.get("x_mitre_platforms", [])),
        ])
        _, kws = _pl("Q " + text)
        technique_keywords[tid.upper()] = set(kws)
 
    # Keyword overlap: scan NVD corpus shards
    mapping: dict[str, set[str]] = defaultdict(set)
    if corpus_dir and os.path.isdir(corpus_dir):
        print(f"[expander] keyword overlap (min_overlap={min_overlap}): "
              f"scanning corpus …", file=sys.stderr)
        for path in sorted(glob.glob(os.path.join(corpus_dir, "nvd_*.txt"))):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    doc_id, tokens = _pl(line)
                    if not doc_id or not cve_pat.match(doc_id):
                        continue
                    doc_set = set(tokens)
                    for tid, kws in technique_keywords.items():
                        if len(doc_set & kws) >= min_overlap:
                            mapping[tid].add(doc_id.upper())
 
    # Write output
    out = {k: sorted(v) for k, v in mapping.items() if v}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
 
    total = sum(len(v) for v in out.values())
    print(f"[expander] wrote {len(out)} techniques → {total:,} CVE pairs → {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # Build technique→CVE map (STIX direct + keyword overlap)
    p_build = sub.add_parser("build-map",
        help="Build technique→CVE map from ATT&CK STIX + NVD keyword overlap")
    p_build.add_argument("--attack",      default="data/attack/enterprise-attack.json")
    p_build.add_argument("--corpus-dir",  default="data/corpus",
        help="NVD corpus shards dir (enables keyword-overlap fallback; recommended)")
    p_build.add_argument("--out",         default="data/index/technique_cve_map.json")
    p_build.add_argument("--min-overlap", default=3, type=int,
        help="Minimum shared keyword count for keyword-based matching (default 3)")

    # Test expansion
    p_test = sub.add_parser("expand", help="Test query expansion")
    p_test.add_argument("--attack",    default="data/attack/enterprise-attack.json")
    p_test.add_argument("--nvd-map",   default="data/index/technique_cve_map.json")
    p_test.add_argument("query", nargs="+")

    args = ap.parse_args()

    if args.cmd == "build-map":
        build_technique_cve_map(
            attack_path = args.attack,
            out_path    = args.out,
            corpus_dir  = args.corpus_dir,
            min_overlap = args.min_overlap,
        )

    elif args.cmd == "expand":
        from nlp import process_line
        expander = QueryExpander(args.attack, args.nvd_map)
        _, terms = process_line("Q " + " ".join(args.query))
        print(f"Original terms : {terms}")
        expanded, weights = expander.expand(terms)
        print(f"Expanded terms : {expanded}")
        print(f"Term weights   : {weights}")
        print()
        print(expander.explain(terms))

    else:
        ap.print_help()