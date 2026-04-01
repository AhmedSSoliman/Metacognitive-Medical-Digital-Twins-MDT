from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1f4e79", fs=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax, start, end):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#334155")
    ax.add_patch(arr)


def make_training_phases(out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.04, 0.35, 0.26, 0.3, "Phase 1: SFT\n\nStructured triple-stream learning\nfrom MIMIC + Medical-O1", fc="#dbeafe", ec="#1d4ed8")
    _box(ax, 0.37, 0.35, 0.26, 0.3, "Phase 2-3: Integrated Training\n\nToM calibration +\nphysiologic trajectory modeling", fc="#e0f2fe", ec="#0369a1")
    _box(ax, 0.70, 0.35, 0.26, 0.3, "Phase 4: GRPO Alignment\n\nMulti-objective reward optimization\n(semantic/meta/empathy/proactivity/safety)", fc="#dcfce7", ec="#166534")

    _arrow(ax, (0.30, 0.50), (0.37, 0.50))
    _arrow(ax, (0.63, 0.50), (0.70, 0.50))

    ax.text(0.5, 0.86, "Training Lifecycle at a Glance", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.14, "Capability first (SFT), then behavior alignment (GRPO)", ha="center", va="center", fontsize=11)

    p = out_dir / "readme_training_lifecycle.png"
    fig.tight_layout()
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_reward_weights(out_dir: Path):
    labels = ["Semantic", "Metacognitive", "Empathy", "Proactivity", "Safety"]
    values = [0.25, 0.20, 0.15, 0.25, 0.15]
    colors = ["#2563eb", "#7c3aed", "#db2777", "#16a34a", "#ea580c"]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bars = ax.bar(labels, [v * 100 for v in values], color=colors, edgecolor="#1f2937")

    ax.set_title("GRPO Reward Weight Composition", fontsize=15, weight="bold")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.2)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{int(v*100)}%", ha="center", va="bottom", fontsize=10)

    ax.text(0.5, -0.16, "Balanced multi-objective optimization in Phase 4", transform=ax.transAxes, ha="center", fontsize=10)

    p = out_dir / "readme_grpo_reward_weights.png"
    fig.tight_layout()
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_soft_cot_flow(out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.05, 0.78, 0.26, 0.14, "Input A: MIMIC examples\n(weak/noisy CoT)", fc="#fde68a", ec="#a16207")
    _box(ax, 0.37, 0.78, 0.26, 0.14, "Input B: Medical-O1 examples\n(gold CoT)", fc="#bbf7d0", ec="#166534")
    _box(ax, 0.69, 0.78, 0.26, 0.14, "Teacher open model\n(Qwen/Llama etc.)", fc="#bfdbfe", ec="#1d4ed8")

    _box(ax, 0.16, 0.54, 0.68, 0.16, "Soft CoT Builder\ncreate/normalize `think_synth`, preserve `think_gold`, tag `think_source` + confidence", fc="#e9d5ff", ec="#6b21a8")

    _box(ax, 0.16, 0.30, 0.30, 0.14, "Quality Filter\nlength + lexical diversity +\nmedical cue coverage", fc="#fecaca", ec="#991b1b")
    _box(ax, 0.54, 0.30, 0.30, 0.14, "Weight Assignment\ngold=1.0\nsynthetic=0.45/0.20", fc="#dcfce7", ec="#166534")

    _box(ax, 0.29, 0.08, 0.42, 0.12, "Weighted SFT Loss + GRPO Alignment\nreliable reasoning without blind hard-mandatory CoT", fc="#dbeafe", ec="#1d4ed8")

    _arrow(ax, (0.18, 0.78), (0.32, 0.70))
    _arrow(ax, (0.50, 0.78), (0.50, 0.70))
    _arrow(ax, (0.82, 0.78), (0.68, 0.70))
    _arrow(ax, (0.50, 0.54), (0.31, 0.44))
    _arrow(ax, (0.50, 0.54), (0.69, 0.44))
    _arrow(ax, (0.31, 0.30), (0.43, 0.20))
    _arrow(ax, (0.69, 0.30), (0.57, 0.20))

    ax.text(0.5, 0.96, "Soft CoT Alignment Decision Flow", ha="center", va="center", fontsize=16, weight="bold")

    p = out_dir / "readme_soft_cot_flow.png"
    fig.tight_layout()
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_training_phases(out_dir)
    make_reward_weights(out_dir)
    make_soft_cot_flow(out_dir)

    print(out_dir / "readme_training_lifecycle.png")
    print(out_dir / "readme_grpo_reward_weights.png")
    print(out_dir / "readme_soft_cot_flow.png")


if __name__ == "__main__":
    main()
