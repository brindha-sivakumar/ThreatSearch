import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#6A4C93", "#F77F00", "#4CC9F0",
    "#B5838D", "#6D6875", "#80B918", "#0077B6", "#E07A5F",
    "#3D405B", "#81B29A", "#F2CC8F", "#118AB2", "#06D6A0",
]

def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ── Loader helpers ────────────────────────────────────────────────────────────

def load_topic_terms(lda_dir: str) -> list[dict]:
    path = os.path.join(lda_dir, "topic_terms.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"topic_terms.json not found at {path}\n"
            "Run: python topic_model.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_doc_topics(lda_dir: str) -> list[dict]:
    path = os.path.join(lda_dir, "doc_topics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"doc_topics.json not found at {path}\n"
            "Run: python topic_model.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lda_model(lda_dir: str):
    """Load saved gensim LDA model and dictionary."""
    try:
        from gensim.models import LdaModel
    except ImportError:
        raise ImportError("gensim not installed. Run: pip install gensim")
    model_path = os.path.join(lda_dir, "lda_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"LDA model not found at {model_path}\n"
            "Run: python topic_model.py"
        )
    return LdaModel.load(model_path)



def plot_wordclouds(topics: list[dict], out_path: str, max_words: int = 40):
    """
    One word cloud per topic. Word size proportional to LDA probability weight.
    Arranged in a grid — layout adapts to number of topics.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        raise ImportError("wordcloud not installed. Run: pip install wordcloud")

    n = len(topics)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 3.2),
                             facecolor="#0F1117")
    axes = np.array(axes).flatten()

    for i, topic in enumerate(topics):
        freq = {w["word"]: w["weight"] for w in topic["terms"][:max_words]}
        color = _color(i)

        wc = WordCloud(
            width=400, height=300,
            background_color="#0F1117",
            color_func=lambda *a, **kw: color,
            max_words=max_words,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(freq)

        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].axis("off")
        axes[i].set_title(
            f"Topic {topic['topic_id']:02d}",
            color=color, fontsize=11, fontweight="bold", pad=4
        )

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("ThreatSearch — CVE Topic Word Clouds",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0F1117")
    plt.close()
    print(f"[viz] word clouds saved → {out_path}")


def plot_tactic_heatmap(
    lda_dir:    str,
    attack_path: str,
    map_path:   str,
    out_path:   str,
):
    """
    Matrix heatmap: ATT&CK tactics (rows) × LDA topics (columns).
    Cell value = number of CVEs that belong to both that tactic and that topic.
    Shows how vulnerability themes align with MITRE kill-chain phases.
    """
    doc_topics = load_doc_topics(lda_dir)
    topics     = load_topic_terms(lda_dir)
    n_topics   = len(topics)

    # Map doc_id → dominant topic
    doc_to_topic = {e["doc_id"].upper(): e["dominant_topic"] for e in doc_topics}

    # Load technique→CVE map
    if not os.path.exists(map_path):
        print(f"[viz] skipping heatmap — {map_path} not found. "
              "Run: python expander.py build-map", file=sys.stderr)
        return

    with open(map_path, encoding="utf-8") as f:
        tech_cve_map = json.load(f)   # {technique_id: [cve_id, ...]}

    # Load ATT&CK tactic names
    with open(attack_path, encoding="utf-8") as f:
        bundle = json.load(f)
    objects = bundle if isinstance(bundle, list) else bundle.get("objects", [])

    # technique_id → list of tactic names
    tech_tactics: dict[str, list[str]] = {}
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id", "").upper()
                phases = [p["phase_name"].replace("-", " ").title()
                          for p in obj.get("kill_chain_phases", [])]
                if tid and phases:
                    tech_tactics[tid] = phases
                break

    # Build tactic set (sorted for consistent axis order)
    all_tactics = sorted({t for phases in tech_tactics.values() for t in phases})
    if not all_tactics:
        print("[viz] no tactic data found — skipping heatmap", file=sys.stderr)
        return

    # Fill matrix: tactic × topic
    matrix = np.zeros((len(all_tactics), n_topics), dtype=int)
    tactic_idx = {t: i for i, t in enumerate(all_tactics)}

    for tid, cve_ids in tech_cve_map.items():
        tactics = tech_tactics.get(tid.upper(), [])
        for cve_id in cve_ids:
            topic = doc_to_topic.get(cve_id.upper())
            if topic is None:
                continue
            for tactic in tactics:
                if tactic in tactic_idx:
                    matrix[tactic_idx[tactic], topic] += 1

    if matrix.sum() == 0:
        print("[viz] heatmap matrix is all zeros — no CVE-tactic overlaps found. "
              "Check that technique_cve_map.json has content.", file=sys.stderr)
        return

    # Short topic labels for x-axis
    topic_labels = [
        f"T{t['topic_id']:02d}: " +
        ", ".join(w["word"] for w in t["terms"][:3])
        for t in topics
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, n_topics * 0.9), max(6, len(all_tactics) * 0.55)),
        facecolor="#0F1117"
    )
    ax.set_facecolor("#0F1117")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    # Axis labels
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels(topic_labels, rotation=45, ha="right",
                       fontsize=7, color="#CCCCCC")
    ax.set_yticks(range(len(all_tactics)))
    ax.set_yticklabels(all_tactics, fontsize=8.5, color="#CCCCCC")

    # Cell annotations (only non-zero)
    for r in range(len(all_tactics)):
        for c in range(n_topics):
            v = matrix[r, c]
            if v > 0:
                text_color = "black" if v > matrix.max() * 0.6 else "white"
                ax.text(c, r, str(v), ha="center", va="center",
                        fontsize=6.5, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#AAAAAA")
    cbar.set_label("CVE count", color="#AAAAAA", fontsize=9)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#AAAAAA")

    ax.set_title("ATT&CK Tactic × Vulnerability Topic Heatmap",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("LDA Topic", color="#AAAAAA", fontsize=9)
    ax.set_ylabel("ATT&CK Tactic", color="#AAAAAA", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    print(f"[viz] tactic heatmap → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ThreatSearch threat cluster visualizations")
    ap.add_argument("--lda-dir",    default="data/lda")
    ap.add_argument("--attack",     default="data/attack/enterprise-attack.json")
    ap.add_argument("--map",        default="data/index/technique_cve_map.json",
                    help="technique→CVE map (for heatmap)")
    ap.add_argument("--out-dir",    default="data/viz")
    ap.add_argument("--max-docs",   default=5000, type=int,
                    help="Max CVEs to plot in scatter (default 5000, more = slower)")
    ap.add_argument("--only",       default=None,
                    choices=["wordclouds", "heatmap"],
                    help="Generate only one visualization instead of all four")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    topics     = load_topic_terms(args.lda_dir)
    doc_topics = load_doc_topics(args.lda_dir)
    print(f"[viz] {len(topics)} topics, {len(doc_topics):,} documents loaded")

    run_all = args.only is None

    if run_all or args.only == "wordclouds":
        plot_wordclouds(
            topics,
            os.path.join(args.out_dir, "topic_wordclouds.png"),
        )

   
    if run_all or args.only == "heatmap":
        plot_tactic_heatmap(
            args.lda_dir,
            args.attack,
            args.map,
            os.path.join(args.out_dir, "tactic_heatmap.png"),
        )

    print(f"\n[viz] all outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
