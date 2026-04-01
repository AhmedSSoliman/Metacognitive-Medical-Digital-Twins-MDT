from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, xy, w, h, text, color="#E8F0FE", edge="#1f4e79", fontsize=10):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def add_arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#2d3748")
    ax.add_patch(arr)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "docs" / "assets" / "soft_cot_alignment_methodology.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Row 1: inputs
    add_box(ax, (0.04, 0.78), 0.23, 0.13, "MIMIC-IV\n(no/weak CoT)", color="#fde68a", edge="#a16207")
    add_box(ax, (0.38, 0.78), 0.23, 0.13, "Medical-O1\n(gold CoT)", color="#bbf7d0", edge="#166534")
    add_box(ax, (0.72, 0.78), 0.23, 0.13, "Teacher Open Model\n(e.g., Qwen3.5-4B)", color="#bfdbfe", edge="#1d4ed8")

    # Row 2: soft mandatory synthesis
    add_box(
        ax,
        (0.19, 0.56),
        0.62,
        0.14,
        "Soft Mandatory CoT Builder\n- Generate/normalize think_synth\n- Preserve think_gold from O1\n- Add think_source + think_confidence",
        color="#e9d5ff",
        edge="#6b21a8",
    )

    # Row 3: quality + weighting
    add_box(
        ax,
        (0.10, 0.34),
        0.35,
        0.14,
        "Quality Filters\nlength, lexical diversity,\nmedical cue coverage, prompt overlap",
        color="#fecaca",
        edge="#991b1b",
    )
    add_box(
        ax,
        (0.55, 0.34),
        0.35,
        0.14,
        "Provenance-Aware Weights\ngold=1.0\nsynthetic(high)=0.45\nsynthetic(low)=0.20",
        color="#d1fae5",
        edge="#065f46",
    )

    # Row 4: training
    add_box(
        ax,
        (0.08, 0.12),
        0.38,
        0.14,
        "SFT (weighted token loss)\ntrain on `think` with sample `think_weight`",
        color="#dbeafe",
        edge="#1e40af",
    )
    add_box(
        ax,
        (0.54, 0.12),
        0.38,
        0.14,
        "GRPO Alignment\noptimize semantic/safety/empathy globally\nreasoning rewards gated by provenance",
        color="#dcfce7",
        edge="#166534",
    )

    # arrows
    add_arrow(ax, (0.16, 0.78), (0.33, 0.70))
    add_arrow(ax, (0.50, 0.78), (0.50, 0.70))
    add_arrow(ax, (0.84, 0.78), (0.67, 0.70))
    add_arrow(ax, (0.50, 0.56), (0.28, 0.48))
    add_arrow(ax, (0.50, 0.56), (0.72, 0.48))
    add_arrow(ax, (0.28, 0.34), (0.27, 0.26))
    add_arrow(ax, (0.72, 0.34), (0.73, 0.26))

    ax.set_title("Soft-Mandatory CoT Alignment with Provenance and Weighting", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
