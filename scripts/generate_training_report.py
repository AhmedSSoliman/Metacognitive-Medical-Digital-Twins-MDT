"""
Generate consolidated SFT + GRPO training report artifacts.

Outputs:
  - summary_metrics.json
  - sft_metrics.csv
  - grpo_metrics.csv (if available)
  - sft_training_curve.png
  - grpo_training_curves.png (if available)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _build_sft_dataframe(sft_dir: Path) -> pd.DataFrame:
    metrics = _read_json(sft_dir / "training_metrics.json")
    history = _read_json(sft_dir / "training_history.json")

    rows = []
    if history.get("training_losses"):
        for idx, loss in enumerate(history["training_losses"], start=1):
            rows.append({"epoch": idx, "train_loss": _safe_float(loss)})

    if not rows and metrics:
        rows = [{
            "epoch": _safe_float(metrics.get("epoch", 0)),
            "train_loss": _safe_float(metrics.get("train_loss", metrics.get("final_loss", 0))),
        }]

    return pd.DataFrame(rows)


def _plot_sft(df: pd.DataFrame, out_file: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2)
    plt.title("SFT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def _build_grpo_dataframe(grpo_dir: Path) -> pd.DataFrame:
    csv_path = grpo_dir / "training_stats.csv"
    json_path = grpo_dir / "training_stats.json"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    stats = _read_json(json_path)
    if not stats:
        return pd.DataFrame()

    return pd.DataFrame(stats)


def _infer_grpo_iterations_from_checkpoints(grpo_dir: Path) -> list[int]:
    """Infer completed GRPO iterations from checkpoint folders like iteration_100."""
    iterations: list[int] = []
    if not grpo_dir.exists():
        return iterations

    for child in grpo_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("iteration_"):
            raw = name.replace("iteration_", "", 1)
            try:
                iterations.append(int(raw))
            except ValueError:
                continue

    return sorted(iterations)


def _plot_grpo(df: pd.DataFrame, out_file: Path) -> None:
    if df.empty or "iterations" not in df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(df["iterations"], df.get("rewards", pd.Series([0] * len(df))))
    axes[0, 0].set_title("Total Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(df["iterations"], df.get("policy_losses", pd.Series([0] * len(df))), color="tab:red")
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(df["iterations"], df.get("kl_divergences", pd.Series([0] * len(df))), color="tab:green")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].grid(alpha=0.25)

    for col, label in [
        ("semantic_rewards", "Semantic"),
        ("metacognitive_rewards", "Metacognitive"),
        ("empathy_rewards", "Empathy"),
        ("proactivity_rewards", "Proactivity"),
        ("safety_rewards", "Safety"),
    ]:
        if col in df.columns:
            axes[1, 1].plot(df["iterations"], df[col], label=label)

    axes[1, 1].set_title("Reward Components")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].grid(alpha=0.25)
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT/GRPO training report")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--report-dir", type=str, default="results/training_reports")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    sft_dir = outputs_dir / "sft"
    grpo_dir = outputs_dir / "grpo"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(args.report_dir) / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    # SFT
    sft_df = _build_sft_dataframe(sft_dir)
    if not sft_df.empty:
        sft_df.to_csv(report_dir / "sft_metrics.csv", index=False)
        _plot_sft(sft_df, report_dir / "sft_training_curve.png")

    # GRPO
    grpo_df = _build_grpo_dataframe(grpo_dir)
    inferred_iterations = _infer_grpo_iterations_from_checkpoints(grpo_dir)
    if not grpo_df.empty:
        grpo_df.to_csv(report_dir / "grpo_metrics.csv", index=False)
        _plot_grpo(grpo_df, report_dir / "grpo_training_curves.png")

    # Summary JSON
    sft_metrics = _read_json(sft_dir / "training_metrics.json")
    grpo_stats = _read_json(grpo_dir / "training_stats.json")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "sft": {
            "final_loss": sft_metrics.get("final_loss", sft_metrics.get("train_loss")),
            "train_runtime": sft_metrics.get("train_runtime"),
            "train_samples_per_second": sft_metrics.get("train_samples_per_second"),
            "train_steps_per_second": sft_metrics.get("train_steps_per_second"),
            "epoch": sft_metrics.get("epoch"),
        },
        "grpo": {
            "iterations_completed": (
                len(grpo_stats.get("iterations", []))
                if grpo_stats and grpo_stats.get("iterations")
                else (max(inferred_iterations) if inferred_iterations else 0)
            ),
            "final_avg_reward": (grpo_stats.get("rewards", [None])[-1] if grpo_stats.get("rewards") else None),
            "final_policy_loss": (grpo_stats.get("policy_losses", [None])[-1] if grpo_stats.get("policy_losses") else None),
            "final_kl_divergence": (grpo_stats.get("kl_divergences", [None])[-1] if grpo_stats.get("kl_divergences") else None),
            "inferred_from_checkpoints": bool(inferred_iterations) and not bool(grpo_stats),
            "checkpoint_iterations": inferred_iterations,
            "note": (
                "Detailed GRPO stats file not found. Iterations inferred from checkpoint folders."
                if bool(inferred_iterations) and not bool(grpo_stats)
                else None
            ),
        }
    }

    with open(report_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Report generated at: {report_dir}")


if __name__ == "__main__":
    main()
