"""Plot results/results.json -> results/agentkv_results.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = [  # fixed categorical order (label, color, marker)
    ("AgentKV", "#2a78d6", "o"),
    ("SnapKV", "#eb6834", "s"),
    ("H2O", "#1baf7a", "^"),
    ("StreamingLLM", "#eda100", "D"),
    ("AgentKV + 4-bit", "#e87ba4", "v"),
]


def main(path="results/results.json"):
    r = json.loads(Path(path).read_text())
    x = [100 * m for m in r["mem_fracs"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    titles = {"single_shot": "Compress once, question already visible\n(how most papers test)",
              "repeated": "Compress after every agent turn\n(how agents actually run)"}
    for ax, setting in zip(axes[:2], ("single_shot", "repeated")):
        ax.set_facecolor(SURFACE)
        ends = []
        for label, color, marker in SERIES:
            y = [100 * a for a in r["settings"][setting][label]]
            ax.plot(x, y, color=color, lw=2, marker=marker, ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
            ends.append([y[-1], label, y[-1]])
        # direct labels at line ends, nudged apart so they never collide
        ends.sort(key=lambda e: -e[0])
        for i in range(1, len(ends)):
            ends[i][0] = min(ends[i][0], ends[i - 1][0] - 6.5)
        for ly, label, y_true in ends:
            ax.annotate(label, (x[-1], y_true), xytext=(x[-1] * 1.12, ly),
                        textcoords="data", va="center", fontsize=9, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}%" for v in x])
        ax.minorticks_off()
        ax.set_ylim(-3, 105)
        ax.set_xlim(x[0] * 0.85, x[-1] * 2.6)
        ax.set_xlabel("KV-cache memory (% of full fp16 cache)")
        ax.set_ylabel("Recall accuracy (%)")
        ax.set_title(titles[setting], fontsize=11, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955),
               ncol=5, frameon=False, fontsize=9)

    # breakdown panel: referenced-before vs never-referenced facts
    ax = axes[2]
    ax.set_facecolor(SURFACE)
    names = ["H2O", "SnapKV", "AgentKV"]
    colors = {"H2O": "#1baf7a", "SnapKV": "#eb6834", "AgentKV": "#2a78d6"}
    bd = r["breakdown"]
    w = 0.36
    for i, n in enumerate(names):
        a = 100 * bd[n]["referenced_before"]
        b = 100 * bd[n]["never_referenced"]
        ax.bar(i - w / 2 - 0.01, a, w, color=colors[n], edgecolor=SURFACE, lw=2, zorder=3)
        ax.bar(i + w / 2 + 0.01, b, w, color=colors[n], edgecolor=colors[n], lw=1,
               hatch="///", alpha=0.45, zorder=3)
        ax.text(i - w / 2, a + 2, f"{a:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, b + 2, f"{b:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_title("Why baselines fail (repeated, 6% memory):\nsolid = fact re-mentioned earlier, hatched = never re-mentioned",
                 fontsize=11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("AgentKV prototype: compression that works in one-shot tests breaks under repeated agent compaction",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, f"Simulation: {r['n_tokens']}-token agent episodes, ~{r['n_facts']:.0f} facts each, "
             f"200 episodes x 8 questions. Full-cache accuracy = {100 * r['full_cache_acc']:.0f}%. "
             "Planted retrieval head in NumPy, not a real LLM.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(path).parent / "agentkv_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
