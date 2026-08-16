"""
training/rollout.py

Phase 4 (Aim 4): rollout-as-a-service -- the parallel rollout worker pool AND
the scaled-training driver loop that consumes from it.

PORTING NOTE (2026-08-12): this file merges TWO source modules:
  - ../Agentic-DT_V1-July/agents/rollout_service.py       -- Part 1 below
  - ../Agentic-DT_V1-July/scripts/run_phase4_scaled_training.py -- Part 2

JUDGMENT CALL on merging the driver in: the target tree allocates exactly one
Phase 4 slot, training/rollout.py, and the two source files are a single
system split only by "library vs entry point". The driver does nothing except
construct a RolloutService, submit RolloutRequests, collect RolloutResults,
and apply the policy-gradient update -- it has no independent identity. It is
merged in as `run_scaled_training()` plus a `main()` argparse entry point, so
`python -m training.rollout` reproduces the original script exactly. The
alternative (a separate scripts/run_phase4.py) would have put a Python driver
under scripts/, which in this layout holds only shell/SBATCH wrappers.

The GRPO math the driver needs (compute_group_relative_advantage,
sequence_logprob, PromptDataset) still comes from training/grpo_utils.py --
deliberately NOT inlined here, for the reason that module's docstring gives.

IMPORT REWRITES (inside rollout_worker, which imports lazily per-process so
each spawned worker builds its own CUDA context):
  - models.multi_stream -> training.backbone
  - training.rewards -> core.rewards
  - hypergraph.verification -> core.hypergraph.verification
  - agents.tool_use -> core.tools.dispatch
  - the sys.path.append hacks are dropped.

ORIGINAL agents/rollout_service.py MODULE DOCSTRING:

    Phase 4: Rollout-as-a-service scaling.

    A minimal, honest implementation of the pattern described in the proposal:
    separate WORKER processes generate rollouts (and execute tool calls) in
    parallel, pushing (prompt, completion, reward_components) tuples onto a
    shared queue that the GRPO training loop consumes in batches.

    This is deliberately built on Python's multiprocessing + a file-based queue
    rather than a bespoke distributed system, so it runs correctly on a single
    HiPerGator node with multiple GPUs without extra infrastructure dependencies.
    For multi-node scaling, replace the queue backend (see `RolloutQueue`) with
    Redis or a similar broker -- the worker/consumer interface stays the same.

ORIGINAL scripts/run_phase4_scaled_training.py MODULE DOCSTRING:

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

# ===========================================================================
# PART 1 -- the rollout worker pool (source: agents/rollout_service.py)
# (original module docstring preserved verbatim above, lines 33-46)
# ===========================================================================

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# CRITICAL: must use "spawn", not the platform default ("fork" on Linux), for
# any multiprocessing that touches CUDA. If the main process has already
# initialized CUDA (which it has here -- run_phase4_scaled_training.py loads
# the trainable policy model onto GPU before calling RolloutService.start()),
# forking a child process afterward is unsafe and commonly causes hangs,
# crashes, or silent corruption in PyTorch/CUDA applications. "spawn" instead
# re-executes a fresh Python interpreter per worker rather than copying the
# CUDA-initialized parent's memory. This must be set before any Process
# objects are created; setting it here (module import time) is the most
# reliable place given multiple entry points (scripts, this module's own
# __main__ block) construct RolloutService.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method can only be set once per interpreter; if something else
    # already set it (e.g. a test harness), just verify it's not "fork"
    current = mp.get_start_method(allow_none=True)
    if current == "fork":
        logger.warning(
            "Multiprocessing start method is 'fork' and could not be changed to 'spawn' "
            "(already set elsewhere). This is unsafe if CUDA has been initialized in the "
            "main process before forking rollout workers -- expect possible hangs or "
            "crashes. Ensure spawn is set before any other multiprocessing code runs."
        )


@dataclass
class RolloutRequest:
    prompt_id: str
    prompt: str
    reference_patient_state: str
    recipient_type: str
    must_mention_facts: list
    # Both optional -- only used to give core.tools.dispatch's get_recent_labs a
    # real per-patient data context (see that module's matching docstring).
    # Without these, the agentic tool-use path still runs correctly, just
    # with get_recent_labs always returning its "not configured" response.
    hadm_id: Optional[int] = None
    prediction_time: Optional[str] = None


@dataclass
class RolloutResult:
    prompt_id: str
    completion: str
    reward_components: dict
    tool_calls_made: int
    generation_latency_ms: float
    worker_id: int


class RolloutQueue:
    """File-system-backed queue (JSONL append + offset tracking) for
    single-node use. Swap this class out for a Redis-backed version if
    scaling across multiple HiPerGator nodes -- the producer/consumer
    interface (`put`, `get_batch`) stays identical.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = mp.Lock()

    def put(self, result: RolloutResult):
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

    def get_batch(self, max_items: int, timeout_s: float = 30.0) -> list[dict]:
        """Reads up to `max_items` lines, then truncates the file to remove
        consumed items. Simple and correct for single-consumer use; not
        safe for multiple concurrent consumers without an added offset file.
        """
        start = time.time()
        while not self.path.exists() and time.time() - start < timeout_s:
            time.sleep(0.5)
        if not self.path.exists():
            return []

        with self._lock:
            lines = self.path.read_text().splitlines()
            batch_lines = lines[:max_items]
            remaining_lines = lines[max_items:]
            self.path.write_text("\n".join(remaining_lines) + ("\n" if remaining_lines else ""))

        return [json.loads(line) for line in batch_lines]


def rollout_worker(
    worker_id: int,
    model_checkpoint: str,
    request_queue: mp.Queue,
    result_queue: RolloutQueue,
    hypergraph_mode: str,
    hypergraph_path: str,
    use_tool_agent: bool,
    group_size: int,
    vitals_labs_path: Optional[str] = None,
):
    """Entry point for a single rollout worker process. Loads its own model
    copy (read-only, no gradient updates happen here) and continuously pulls
    prompts from `request_queue`, generates + scores completions, and pushes
    results to `result_queue`.

    `vitals_labs_path`: optional path to a combined [hadm_id, charttime,
    variable, value] parquet (the same format core/hypergraph/construction.py's
    Phase 3 pipeline builds, e.g. cache/mimic/derivation_timeseries.parquet)
    -- gives core.tools.dispatch's get_recent_labs a real per-patient data
    context to query. Loaded ONCE per worker and grouped by hadm_id for
    fast per-request slicing; omit to run with get_recent_labs always
    returning its "not configured" response.
    """
    import torch
    from training.backbone import MultiStreamModel, MultiStreamConfig
    from core.rewards import compute_total_reward, RewardWeights
    from core.hypergraph.verification import InterimRuleBasedChecker, LearnedHypergraphChecker
    from core.tools.dispatch import make_default_registry, run_agentic_turn

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"worker-{worker_id}")
    logger.info("Worker %d starting, loading model from %s", worker_id, model_checkpoint)

    ts_model = MultiStreamModel(MultiStreamConfig(base_model_name=model_checkpoint))
    ts_model.model.eval()

    if hypergraph_mode == "interim":
        checker = InterimRuleBasedChecker()
    else:
        checker = LearnedHypergraphChecker(hypergraph_path)

    patient_timeseries_by_hadm = None
    if use_tool_agent and vitals_labs_path:
        import pandas as pd
        ts_df = pd.read_parquet(vitals_labs_path)
        patient_timeseries_by_hadm = {hadm_id: group for hadm_id, group in ts_df.groupby("hadm_id")}
        logger.info("Worker %d loaded per-patient timeseries for %d admissions from %s",
                    worker_id, len(patient_timeseries_by_hadm), vitals_labs_path)

    weights = RewardWeights()

    while True:
        try:
            req_dict = request_queue.get(timeout=5.0)
        except queue.Empty:
            continue
        if req_dict is None:  # poison pill -- shutdown signal
            logger.info("Worker %d received shutdown signal.", worker_id)
            break

        req = RolloutRequest(**req_dict)
        start = time.time()

        if use_tool_agent:
            # Built PER-REQUEST (not once at worker startup) so get_recent_labs
            # is bound to THIS request's specific patient -- a shared, worker-
            # level registry could not know which patient a given tool call
            # was about, since requests for many different patients cycle
            # through the same worker process over its lifetime.
            patient_ts = (patient_timeseries_by_hadm.get(req.hadm_id)
                          if patient_timeseries_by_hadm is not None and req.hadm_id is not None else None)
            tool_registry = make_default_registry(
                checker, patient_timeseries=patient_ts, prediction_time_cutoff=req.prediction_time,
            )

            def gen_fn(p):
                return ts_model.generate(p, num_return_sequences=1)[0]
            agent_result = run_agentic_turn(gen_fn, req.prompt, tool_registry)
            completions = [agent_result["final_generation"]]
            tool_calls_made = len(agent_result["tool_calls"])
        else:
            completions = ts_model.generate(req.prompt, num_return_sequences=group_size)
            tool_calls_made = 0

        latency_ms = (time.time() - start) * 1000

        for completion in completions:
            reward_dict = compute_total_reward(
                generated_text=completion,
                reference_patient_state=req.reference_patient_state,
                hypergraph_checker=checker,
                recipient_type=req.recipient_type,
                must_mention_facts=req.must_mention_facts,
                weights=weights,
            )
            result = RolloutResult(
                prompt_id=req.prompt_id,
                completion=completion,
                reward_components=reward_dict,
                tool_calls_made=tool_calls_made,
                generation_latency_ms=latency_ms,
                worker_id=worker_id,
            )
            result_queue.put(result)


class RolloutService:
    """Spawns N worker processes and manages the shared request/result queues."""

    def __init__(self, num_workers: int, model_checkpoint: str, result_queue_path: str,
                 hypergraph_mode: str = "interim", hypergraph_path: str = "",
                 use_tool_agent: bool = False, group_size: int = 8,
                 vitals_labs_path: Optional[str] = None):
        self.num_workers = num_workers
        self.request_queue = mp.Queue()
        self.result_queue = RolloutQueue(result_queue_path)
        self.workers = []
        self._start_args = (model_checkpoint, hypergraph_mode, hypergraph_path, use_tool_agent,
                             group_size, vitals_labs_path)

    def start(self):
        (model_checkpoint, hypergraph_mode, hypergraph_path, use_tool_agent,
         group_size, vitals_labs_path) = self._start_args
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=rollout_worker,
                args=(worker_id, model_checkpoint, self.request_queue, self.result_queue,
                      hypergraph_mode, hypergraph_path, use_tool_agent, group_size, vitals_labs_path),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
        logger.info("Started %d rollout workers.", self.num_workers)

    def submit(self, req: RolloutRequest):
        self.request_queue.put(asdict(req))

    def collect(self, max_items: int, timeout_s: float = 60.0) -> list[dict]:
        return self.result_queue.get_batch(max_items, timeout_s=timeout_s)

    def shutdown(self):
        for _ in self.workers:
            self.request_queue.put(None)
        for p in self.workers:
            p.join(timeout=30)
        logger.info("Rollout service shut down.")



def _queue_smoke_test():
    """Original agents/rollout_service.py __main__ smoke test, preserved as a
    callable so this file can have a single real __main__ (the Phase 4 driver).
    Run with: python -m training.rollout --queue_smoke_test
    """
    # Throughput/latency smoke test using the queue mechanics only (no model load)
    import tempfile
    tmp_path = tempfile.mktemp(suffix=".jsonl")
    rq = RolloutQueue(tmp_path)

    dummy_result = RolloutResult(
        prompt_id="p1", completion="dummy", reward_components={"total": 0.7},
        tool_calls_made=0, generation_latency_ms=120.0, worker_id=0,
    )
    for _ in range(10):
        rq.put(dummy_result)

    batch = rq.get_batch(max_items=5)
    print(f"Retrieved {len(batch)} items from queue (expected 5)")
    remaining = rq.get_batch(max_items=100)
    print(f"Retrieved remaining {len(remaining)} items (expected 5)")
    os.remove(tmp_path)
    print("Queue mechanics smoke test passed.")


# ===========================================================================
# PART 2 -- the Phase 4 scaled-training driver
# (source: scripts/run_phase4_scaled_training.py)
# ===========================================================================

# Part 2's own imports. json/logging/time/Path are already imported at the
# top of Part 1; argparse/defaultdict/numpy/torch/AdamW are new here. Kept at
# this point in the file (rather than hoisted to the very top) so that
# importing this module for its Part 1 queue classes alone -- as
# tests/test_training/test_rollout_queue.py does -- is unaffected by them...
# except that Python executes module bodies top-to-bottom, so these DO run on
# any import of training.rollout. numpy/torch are therefore hard import
# requirements of this module, exactly as they were for the original driver
# script. The test guards with pytest.importorskip accordingly.
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.optim import AdamW

from training.backbone import MultiStreamModel, MultiStreamConfig
# RolloutService / RolloutRequest are defined in Part 1 of THIS file.
from training.grpo_utils import compute_group_relative_advantage, sequence_logprob, PromptDataset

# The original driver script configured logging at import; preserved here so
# `python -m training.rollout` behaves identically to the original
# `python scripts/run_phase4_scaled_training.py`.
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2_checkpoint", required=True,
                         help="Path to the Phase 2 GRPO-aligned checkpoint to continue training from.")
    parser.add_argument("--hypergraph_path", required=True,
                         help="Path to a CLINICALLY REVIEWED hypergraph JSON (status must be "
                              "CLINICALLY_REVIEWED) -- see core/hypergraph/verification.py, which will "
                              "refuse to load an unreviewed one.")
    parser.add_argument("--prompt_dataset", required=True,
                         help="Output of core/cohort/grpo_prompts.py.")
    parser.add_argument("--num_rollout_workers", type=int, default=3,
                         help="Number of parallel worker PROCESSES generating rollouts. This, "
                              "along with --group_size, is the main throughput/cost knob Manuscript 4 "
                              "should sweep across to build its engineering-trade-offs tables.")
    parser.add_argument("--group_size", type=int, default=8,
                         help="Completions sampled per prompt for the group-relative advantage "
                              "computation. Ignored (effectively 1) when --use_tool_agent is set, "
                              "since the agentic tool-use loop currently runs one reasoning trace "
                              "per prompt, not a sampled group -- see Part 1 of this module's "
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
                         help="Enable ReAct-style mid-reasoning tool calls (core/tools/dispatch.py) inside "
                              "each rollout, instead of plain sampled generation with no tool access.")
    parser.add_argument("--vitals_labs_path", default="",
                         help="Path to a combined [hadm_id, charttime, variable, value] parquet (same "
                              "format core/hypergraph/construction.py's Phase 3 pipeline builds, e.g. "
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
    # Part 1 of this module's module docstring for why "spawn" (not the
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
    import sys as _sys
    if "--queue_smoke_test" in _sys.argv:
        # Original agents/rollout_service.py __main__ behavior.
        _queue_smoke_test()
    else:
        # Original scripts/run_phase4_scaled_training.py __main__ behavior.
        main()
