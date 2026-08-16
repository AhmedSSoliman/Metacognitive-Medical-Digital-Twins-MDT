"""
scripts/run_phase4_scaled_training.py

Phase 4 orchestration: ties agents.rollout_service (parallel rollout workers,
optionally using agentic tool use) to a GRPO-style policy update loop that
consumes batches of scored rollouts from the shared queue instead of
generating them synchronously in-process (contrast with training/grpo_trainer.py,
Phase 2's simpler single-process version, which since a later refactor uses
TRL's own GRPOTrainer directly rather than the hand-rolled loop used here).

This is the "engineering trade-offs" system Manuscript 4 is meant to report on:
run this with different --num_rollout_workers and --group_size settings and
log the resulting throughput/latency to build that manuscript's Table 1/2/3.

IMPORT NOTE (a real bug caught and fixed): this script originally imported
`compute_group_relative_advantage`, `sequence_logprob`, and `PromptDataset`
directly from training/grpo_trainer.py. When that file was refactored to use
TRL's GRPOTrainer (which handles those mechanics internally), those three
names stopped being defined there at all -- silently breaking this script's
import with an ImportError. They now live in training/grpo_utils.py, a
small shared module kept independent of whichever approach Phase 2 happens
to use, specifically so refactoring Phase 2 again in the future can't
silently break Phase 4 the same way twice.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.multi_stream import MultiStreamModel, MultiStreamConfig
from agents.rollout_service import RolloutService, RolloutRequest
from training.grpo_utils import compute_group_relative_advantage, sequence_logprob, PromptDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_checkpoint", required=True,
                         help="Path to the Phase 2 GRPO-aligned checkpoint to continue training from.")
    parser.add_argument("--hypergraph_path", required=True,
                         help="Path to a CLINICALLY REVIEWED hypergraph JSON (status must be "
                              "CLINICALLY_REVIEWED) -- see hypergraph/verification.py, which will "
                              "refuse to load an unreviewed one.")
    parser.add_argument("--prompt_dataset", required=True,
                         help="Output of data/build_grpo_prompts.py.")
    parser.add_argument("--num_rollout_workers", type=int, default=3,
                         help="Number of parallel worker PROCESSES generating rollouts. This, "
                              "along with --group_size, is the main throughput/cost knob Manuscript 4 "
                              "should sweep across to build its engineering-trade-offs tables.")
    parser.add_argument("--group_size", type=int, default=8,
                         help="Completions sampled per prompt for the group-relative advantage "
                              "computation. Ignored (effectively 1) when --use_tool_agent is set, "
                              "since the agentic tool-use loop currently runs one reasoning trace "
                              "per prompt, not a sampled group -- see agents/rollout_service.py's "
                              "rollout_worker for exactly where this branches.")
    parser.add_argument("--num_iterations", type=int, default=3000)
    parser.add_argument("--prompts_per_batch", type=int, default=8)
    parser.add_argument("--kl_coef", type=float, default=0.02,
                         help="Weight on the KL-to-reference-policy penalty term. Higher values keep "
                              "the policy closer to the Phase 2 checkpoint (safer, slower to improve); "
                              "lower values allow faster movement (riskier, more prone to drifting into "
                              "degenerate behavior the reward function doesn't actually penalize).")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                         help="Deliberately much smaller than Phase 1/2's learning rates -- Phase 4 is "
                              "CONTINUING an already-aligned policy at scale, not training from scratch, "
                              "so large updates risk undoing Phase 2's alignment work rather than "
                              "refining it.")
    parser.add_argument("--use_tool_agent", action="store_true",
                         help="Enable ReAct-style mid-reasoning tool calls (agents/tool_use.py) inside "
                              "each rollout, instead of plain sampled generation with no tool access.")
    parser.add_argument("--vitals_labs_path", default="",
                         help="Path to a combined [hadm_id, charttime, variable, value] parquet (same "
                              "format hypergraph/construction.py's Phase 3 pipeline builds, e.g. "
                              "cache/mimic/derivation_timeseries.parquet) -- gives --use_tool_agent's "
                              "get_recent_labs tool a real per-patient data context. Ignored if "
                              "--use_tool_agent is not set; get_recent_labs falls back to a clear "
                              "'not configured' response if this is left empty.")
    parser.add_argument("--output_dir", default="./checkpoints/phase4_scaled")
    parser.add_argument("--throughput_log_path", default="./logs/phase4_throughput.jsonl",
                         help="Appended to (not overwritten) across runs, so multiple sweeps over "
                              "different --num_rollout_workers / --group_size settings accumulate "
                              "into one comparable log for Manuscript 4's tables.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.throughput_log_path).parent.mkdir(parents=True, exist_ok=True)

    # Trainable policy lives in THIS (main) process; rollout WORKER processes
    # (started below via RolloutService) hold their own separate, read-only
    # copies of the model loaded fresh from args.phase2_checkpoint. This
    # split exists because generation (rollout workers) and the actual
    # gradient-update step (this process) have very different resource
    # profiles -- workers benefit from being spread across multiple GPUs /
    # processes for parallel throughput, while the single optimizer step
    # needs to see all of a batch's results together before it can update.
    ts_model = MultiStreamModel(MultiStreamConfig(base_model_name=args.phase2_checkpoint))
    model = ts_model.model
    tokenizer = ts_model.tokenizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Frozen reference copy for the KL penalty -- a SEPARATE model instance
    # from `model` above (not the same object with gradients disabled),
    # since its weights must stay fixed at the Phase 2 checkpoint's values
    # throughout Phase 4 training while `model`'s weights keep updating.
    ref_model = MultiStreamModel(MultiStreamConfig(base_model_name=args.phase2_checkpoint)).model
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Spawns args.num_rollout_workers separate PROCESSES (not threads) that
    # each load their own copy of the model and continuously pull prompts
    # off a shared queue, generate + score completions, and push results to
    # a second shared queue this main process reads from below. See
    # agents/rollout_service.py's module docstring for why "spawn" (not the
    # platform-default "fork") is forced for this multiprocessing -- forking
    # after this main process has already initialized CUDA (which it just
    # did, loading `model` and `ref_model` above) is a well-documented
    # PyTorch/CUDA hang-and-crash risk.
    service = RolloutService(
        num_workers=args.num_rollout_workers,
        model_checkpoint=args.phase2_checkpoint,
        result_queue_path="./cache/rollout_results.jsonl",
        hypergraph_mode="learned",  # Phase 4 always uses the reviewed hypergraph, never the interim rule-based fallback
        hypergraph_path=args.hypergraph_path,
        use_tool_agent=args.use_tool_agent,
        group_size=args.group_size,
        vitals_labs_path=args.vitals_labs_path or None,
    )
    service.start()

    dataset = PromptDataset(args.prompt_dataset)

    throughput_log = open(args.throughput_log_path, "a")
    step = 0
    rng = np.random.default_rng(42)  # fixed seed: makes which prompts get sampled at which step reproducible across runs

    try:
        while step < args.num_iterations:
            iter_start = time.time()

            # Submit a batch of prompts to the rollout workers. Sampling
            # WITH replacement (np.random.Generator.integers default) is
            # intentional here -- unlike a standard SFT epoch, GRPO doesn't
            # need to see every example exactly once per "epoch"; it needs
            # enough diverse (prompt, group-of-completions) pairs over many
            # steps, and simple random sampling is sufficient for that.
            batch_indices = rng.integers(0, len(dataset), size=args.prompts_per_batch)
            for idx in batch_indices:
                example = dataset[int(idx)]
                req = RolloutRequest(
                    # Encodes both the step and the dataset index into the ID
                    # so results can be matched back to their originating
                    # prompt after coming back through the (order-scrambling,
                    # multi-worker) results queue -- see the `next(...)`
                    # lookup below where this ID is parsed back out.
                    prompt_id=f"step{step}_idx{idx}",
                    prompt=example["prompt"],
                    reference_patient_state=example["reference_patient_state"],
                    recipient_type=example["recipient_type"],
                    must_mention_facts=example["must_mention_facts"],
                    hadm_id=example["hadm_id"],
                    prediction_time=example["prediction_time"],
                )
                service.submit(req)

            # Collect results. With --use_tool_agent, each prompt produces
            # exactly ONE result (the agentic loop's final answer after
            # however many tool-call rounds it took), not a sampled group --
            # hence expected_results uses group_size only in the non-agentic
            # case.
            expected_results = args.prompts_per_batch * (args.group_size if not args.use_tool_agent else 1)
            results = service.collect(max_items=expected_results, timeout_s=120.0)

            if len(results) < expected_results:
                # NOT a fatal error -- workers may simply be slower than the
                # 120s collection timeout for this particular batch. Logged
                # as a warning (not raised) so training can continue with a
                # partial batch; a persistently low collection rate across
                # many steps is itself the throughput signal Manuscript 4 is
                # meant to measure and report, not something to paper over.
                logger.warning(
                    "Step %d: expected %d rollout results, got %d (workers may be falling "
                    "behind -- this is exactly the throughput bottleneck Manuscript 4 should measure).",
                    step, expected_results, len(results),
                )

            # Group results by prompt_id so each prompt's own group of
            # completions gets its own group-relative advantage computation
            # -- results arrive interleaved across different prompts and
            # workers, in no particular order, so this regrouping step is
            # necessary before GRPO's per-group math can be applied.
            grouped = defaultdict(list)
            for r in results:
                grouped[r["prompt_id"]].append(r)

            total_loss = 0.0
            n_updates = 0
            for prompt_id, group_results in grouped.items():
                rewards = np.array([r["reward_components"]["total"] for r in group_results])
                if len(rewards) < 2:
                    # A group-relative advantage is meaningless (division by
                    # a std of a single point, or comparing a value to
                    # itself) with fewer than 2 completions -- skip rather
                    # than compute a degenerate/undefined advantage for
                    # this prompt on this step.
                    continue
                advantages = compute_group_relative_advantage(rewards)

                # Recover the original prompt TEXT (not just its ID) to
                # recompute log-probabilities under the CURRENT policy --
                # the rollout worker that generated this completion may have
                # been using a slightly stale copy of the model (workers
                # aren't updated every single step), so log-probs must be
                # recomputed fresh here under `model` as it exists RIGHT NOW,
                # not reused from whatever the worker itself may have logged.
                original_prompt = next(
                    dataset[int(i)]["prompt"] for i in batch_indices
                    if f"step{step}_idx{i}" == prompt_id
                )
                formatted_prompt = ts_model.format_prompt(original_prompt)

                for r, advantage in zip(group_results, advantages):
                    policy_logprob = sequence_logprob(model, tokenizer, formatted_prompt, r["completion"],
                                                       next(model.parameters()).device)
                    with torch.no_grad():
                        ref_logprob = sequence_logprob(ref_model, tokenizer, formatted_prompt, r["completion"],
                                                         next(model.parameters()).device)
                    kl = policy_logprob - ref_logprob
                    # Standard GRPO/PPO-style objective: push probability mass
                    # toward higher-advantage completions (the negative sign
                    # turns this into something an optimizer MINIMIZES, since
                    # we want to maximize advantage-weighted log-prob), while
                    # the KL term discourages drifting too far from the
                    # reference policy in a single step.
                    loss = -(float(advantage) * policy_logprob) + args.kl_coef * kl
                    total_loss = total_loss + loss
                    n_updates += 1

            if n_updates > 0:
                # Averaging by n_updates (not just dividing by a fixed batch
                # size) matters because n_updates can vary batch-to-batch --
                # some prompts may have been skipped above (the < 2 rewards
                # case), or fewer results than expected may have come back
                # from a slow batch; averaging by however many updates
                # actually happened keeps the loss scale comparable across
                # batches of different effective sizes.
                total_loss = total_loss / n_updates
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            iter_latency_s = time.time() - iter_start
            tool_call_latencies = [r["generation_latency_ms"] for r in results]

            # This whole record is the raw data Manuscript 4's throughput
            # and latency tables/figures are built from -- see the module
            # docstring above and the manuscript's own outline
            # (Manuscript4_Outline_and_Plan.md) for exactly which of these
            # fields map to which planned table/figure.
            throughput_record = {
                "step": step,
                "n_results_collected": len(results),
                "n_expected": expected_results,
                "iteration_latency_s": iter_latency_s,
                "rollouts_per_second": len(results) / max(iter_latency_s, 1e-6),
                "mean_reward": float(np.mean([r["reward_components"]["total"] for r in results])) if results else None,
                "gen_latency_p50_ms": float(np.percentile(tool_call_latencies, 50)) if tool_call_latencies else None,
                "gen_latency_p95_ms": float(np.percentile(tool_call_latencies, 95)) if tool_call_latencies else None,
                "mean_tool_calls_per_rollout": float(np.mean([r["tool_calls_made"] for r in results])) if results else None,
            }
            # JSONL (one JSON object per line), not a single JSON array, so
            # the log file remains valid/readable even if a training run is
            # killed mid-way -- an array would be left syntactically broken
            # (missing its closing bracket) by an abrupt interruption.
            throughput_log.write(json.dumps(throughput_record) + "\n")
            throughput_log.flush()

            if step % 10 == 0:
                logger.info("Step %d: %s", step, throughput_record)

            if step % 100 == 0 and step > 0:
                ckpt_path = Path(args.output_dir) / f"checkpoint-{step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            step += 1
    finally:
        # `finally` (not just code after the loop) ensures the rollout worker
        # processes are always told to shut down and the log file handle is
        # always closed, even if training is interrupted (Ctrl-C, a SLURM
        # time limit, an unhandled exception mid-loop) -- leaving orphaned
        # worker processes running, or an unflushed/unclosed log file, would
        # otherwise be an easy way to leak GPU memory or lose the last few
        # steps of throughput data.
        service.shutdown()
        throughput_log.close()

    final_path = Path(args.output_dir) / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Phase 4 scaled training complete. Throughput log: %s", args.throughput_log_path)


if __name__ == "__main__":
    main()
