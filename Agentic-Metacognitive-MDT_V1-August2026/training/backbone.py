"""
training/backbone.py

Backbone loading, LoRA configuration, backbone selection, generation, and the
multi-stream model wrapper.

PORTING NOTE (2026-08-12): ported from the source repo's
models/multi_stream.py (../Agentic-DT_V1-July/models/multi_stream.py), with
its intra-project imports rewritten (see below). This is the ONLY file in the
port that consolidates into training/backbone.py.

WHAT COUNTS AS "BACKBONE" HERE, AND WHY THE WHOLE FILE CAME ACROSS -- the
judgment call, stated plainly. The target tree describes backbone.py as
"Model loading, LoRA config, backbone selection". Reading multi_stream.py in
full, it contains five things:

  1. MultiStreamConfig -- backbone id + LoRA r/alpha/dropout/target_modules +
     4bit/seq-len/device_map. Unambiguously backbone config. STAYS.
  2. MultiStreamModel.__init__ -- the actual loading logic, including the
     adapter-checkpoint-vs-base-model branch and the Unsloth-mirror
     workaround. Unambiguously backbone loading, and it carries some of the
     most expensive debugging history in the project. STAYS.
  3. format_prompt / generate / _generate_once / _StopOnClosingTag -- prompt
     templating and generation, with the four chat-template fallback variants
     and the malformed-generation retry loop. Arguably "inference" rather than
     "backbone", but they are METHODS ON the loaded model object and depend on
     its tokenizer; separating them would mean either splitting a class across
     files or exporting the tokenizer, both worse than keeping them together.
     STAYS, flagged here as the ambiguous case.
  4. ForecastHead + extract_pooled_hidden_state -- an optional auxiliary
     REGRESSION head, not part of the backbone or the generative path at all.
     Kept here (it is small, torch-native, and has no other natural home in
     the target tree) but noted explicitly as a SEPARATE CONCERN: if this
     grows, it belongs in its own module. It is the intended producer of the
     `risk_score` column evaluation/predictive.py consumes.
  5. ConversationSession -- multi-turn chat state on top of the model.
     Application-layer, not backbone. Kept here to avoid stranding it, and
     because the Streamlit/eval callers in the source repo imported it from
     this same module. Noted as a separate concern.

NOTE ON "multi-stream fusion": there is no fusion logic to place anywhere.
The source docstring below is explicit that this module does NOT invent an
attention mechanism -- stream separation is enforced by the prompt contract
plus the parser (core/parsing.py), not by any architectural change.

IMPORT REWRITES: `from models.stream_parsing import (...)` ->
`from core.schema import STREAM_TAGS, FORECAST_FORMAT_EXAMPLE,
STREAM_SYSTEM_PROMPT` and `from core.parsing import ForecastEntry,
parse_forecast_text, ParsedStreams, parse_streams`. The re-export comment is
preserved and updated below.

ORIGINAL models/multi_stream.py MODULE DOCSTRING:

    Multi-Stream Cognitive Architecture (Aim 1 / Phase 1).

    Wraps a causal LLM backbone (Nemotron-Mini, MedGemma, GPT-OSS, or similar) so
    that a single forward pass produces four structurally tagged, concurrently
    generated streams (named "Multi-Stream", not "Triple-Stream", precisely
    because it's four, not three):

        <think>          internal deliberation / reasoning trace
        <patient_state>  structured summary of current physiological state
        <forecast>       quantitative near-term trajectory prediction
        <user_belief>    model's estimate of the recipient's knowledge/emotional state

    This module does NOT invent a new attention mechanism -- the "structural
    separation" is enforced via (a) a constrained generation grammar (stream tags
    must appear in a fixed order and be well-formed) and (b) a parser that
    extracts each stream's content for separate reward computation in Phase 2.
    Keeping this simple and inspectable is intentional: the reward-shaping and
    hypergraph-verification logic downstream depends on being able to reliably
    isolate each stream's text.
"""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, get_peft_model, TaskType

# STREAM_TAGS, STREAM_SYSTEM_PROMPT, ParsedStreams, parse_streams, and
# parse_forecast_text live in core/schema.py and core/parsing.py, which have
# zero torch/transformers/peft dependencies -- imported here and re-exported
# so every caller that did `from models.multi_stream import parse_streams`
# (etc.) keeps working after the rename to this module. See core/schema.py's
# preserved docstring for why that split was made in the first place.
from core.schema import STREAM_TAGS, FORECAST_FORMAT_EXAMPLE, STREAM_SYSTEM_PROMPT
from core.parsing import ForecastEntry, parse_forecast_text, ParsedStreams, parse_streams

logger = logging.getLogger(__name__)


class _StopOnClosingTag(StoppingCriteria):
    """Halts generation as soon as `</user_belief>` (the last of the four
    streams) has been emitted, per sequence in the batch, rather than
    relying on the model's own learned end-of-sequence behavior.

    Added after a real, confirmed generation-termination regression: two
    independently trained Phase 1 checkpoints (with and without a
    since-ruled-out suspected cause) both failed to stop cleanly in 12/12
    tested (prompt x decoding-config) combinations -- either looping
    indefinitely inside <think> or repeating the entire well-formed
    response multiple times. In every case where a well-formed response
    WAS produced, the correct answer was present in full the first time;
    the model just kept generating past it. Rather than keep spending
    GPU-hours chasing the root cause of the model's own EOS unreliability,
    this stops generation the moment a complete, correct response exists,
    which directly fixes the practical usability problem regardless of
    why the underlying learned behavior is unreliable.

    Checks by decoding a small trailing token window (not exact token-ID
    matching), since tokenization boundaries for a literal string can shift
    depending on preceding context -- the same category of issue as this
    project's `sequence_logprob` tokenization-boundary bug (see README bug
    log) -- and substring matching on decoded text sidesteps it entirely.
    """

    def __init__(self, tokenizer, closing_tag: str = "</user_belief>", check_last_n_tokens: int = 12):
        self.tokenizer = tokenizer
        self.closing_tag = closing_tag
        self.check_last_n_tokens = check_last_n_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i in range(input_ids.shape[0]):
            tail_ids = input_ids[i, -self.check_last_n_tokens:]
            tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
            if self.closing_tag in tail_text:
                done[i] = True
        return done


@dataclass
class MultiStreamConfig:
    base_model_name: str  # e.g. "nvidia/Nemotron-Mini-4B-Instruct", "google/medgemma-4b-it"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    load_in_4bit: bool = True
    max_seq_len: int = 4096
    device_map: str = "auto"


class ForecastHead(nn.Module):
    """Optional lightweight regression head for quantitative forecasting,
    complementing (not replacing) the text-based <forecast> stream.

    Rationale: free-text numeric generation from LLMs is known to be
    unreliable for precise quantitative prediction (the model can "say" a
    plausible-looking number without it being well-calibrated). A small
    regression head trained on the backbone's own hidden states, with a
    real regression loss (MSE against true future values), is a more
    statistically principled way to get a genuine risk/forecast score --
    e.g. exactly the `risk_score` column evaluation/predictive.py's
    Deterioration Detection endpoint requires and which nothing else in
    this codebase actually produces.

    This is intentionally a SEPARATE, small module (not fused into the
    generative loss) so it can be trained with straightforward supervised
    regression on labeled MIMIC-IV outcomes (core/cohort/mimic.py already
    produces deterioration_6h and related labels), independent of whatever
    is happening with the generative streams and GRPO alignment.
    """

    def __init__(self, hidden_size: int, num_targets: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_targets),
        )

    def forward(self, pooled_hidden_state: torch.Tensor) -> torch.Tensor:
        """`pooled_hidden_state` should be a single pooled representation per
        example (e.g. the last non-padding token's hidden state, or a mean-
        pooled representation) -- shape (batch, hidden_size). Returns
        (batch, num_targets) raw regression outputs.
        """
        return self.net(pooled_hidden_state)


def extract_pooled_hidden_state(model_outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pools the LAST hidden layer over non-padding tokens. Requires the
    forward pass to have been run with `output_hidden_states=True`.
    """
    last_hidden = model_outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (batch, seq_len, 1)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class MultiStreamModel(nn.Module):
    """LoRA-wrapped causal LM with stream-parsing utilities.

    This class deliberately does NOT modify the backbone's architecture.
    Structural separation is a property of the TRAINING DATA / GENERATION
    FORMAT (see STREAM_SYSTEM_PROMPT), not a novel layer. This keeps the
    approach compatible with any off-the-shelf instruction-tuned backbone
    and makes the "which base model performs best" comparison in Phase 1
    an honest apples-to-apples test.
    """

    def __init__(self, cfg: MultiStreamConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            cfg.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_kwargs = {}
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # cfg.base_model_name is used two different ways by callers of this
        # class: a real HF backbone id/path for Phase 1 SFT-from-scratch
        # (e.g. "google/medgemma-4b-it"), or a Phase 1/Phase 2 TRAINED
        # checkpoint directory (e.g. "./checkpoints/phase2_grpo/final") for
        # evaluation, the Streamlit demo, the rollout service, and Phase 4.
        # Every one of those trained-checkpoint callers was silently broken
        # before this fix: a checkpoint dir has no config.json (only
        # adapter_config.json + adapter weights), so plain
        # AutoModelForCausalLM couldn't load it at all -- and even if it
        # could, the code below unconditionally built a FRESH, randomly-
        # initialized LoRA via get_peft_model(), discarding whatever trained
        # weights were actually being asked for.
        adapter_config_path = Path(cfg.base_model_name) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            underlying_base_model = adapter_cfg["base_model_name_or_path"]
            logger.info(
                "%s is a trained adapter checkpoint (base model: %s) -- loading the "
                "REAL trained weights.", cfg.base_model_name, underlying_base_model,
            )
            # First attempt (plain transformers.AutoModelForCausalLM +
            # peft.PeftModel.from_pretrained) failed with "Cannot copy out
            # of meta tensor; no data!" even after forcing an explicit
            # single-device map -- root cause is that Phase 1/2 training
            # used Unsloth, which silently substitutes the requested base
            # model (e.g. "google/medgemma-4b-it") for its OWN pre-
            # quantized 4-bit mirror (here:
            # "unsloth/medgemma-4b-it-unsloth-bnb-4bit", recorded as
            # base_model_name_or_path in adapter_config.json) for faster
            # loading. That mirror's quantized weights are only
            # materializable through Unsloth's own loader -- confirmed by
            # scripts/validate_phase1_checkpoint.py, the one place in this
            # project that has always loaded these checkpoints successfully,
            # which uses exactly the pattern below. Fighting plain HF/PEFT
            # into working with an Unsloth-specific checkpoint isn't worth
            # it when Unsloth's own loader already does this correctly.
            try:
                from unsloth import FastLanguageModel
            except ImportError as e:
                raise ImportError(
                    f"Unsloth is required to load this checkpoint: its base model "
                    f"({underlying_base_model}) is one of Unsloth's own pre-quantized "
                    f"4-bit mirrors, which plain transformers cannot materialize "
                    f"correctly. pip install unsloth."
                ) from e

            # Same fix as training/grpo.py's load_model_and_tokenizer_unsloth:
            # Unsloth's ahead-of-time compiled LoRA forward
            # (unsloth_compiled_cache/Linear_peft_forward.py, regenerated fresh
            # on every cold model load) references peft's VARIANT_KWARG_KEYS
            # without importing it -- confirmed via a real crash here
            # (NameError: name 'VARIANT_KWARG_KEYS' is not defined). Injecting
            # into builtins survives cache regeneration; patching the
            # generated file does not.
            import builtins
            from peft.tuners.lora.layer import VARIANT_KWARG_KEYS as _VARIANT_KWARG_KEYS
            builtins.VARIANT_KWARG_KEYS = _VARIANT_KWARG_KEYS

            unsloth_model, unsloth_tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.base_model_name,
                max_seq_length=cfg.max_seq_len,
                dtype=None,
                load_in_4bit=cfg.load_in_4bit,
                fast_inference=False,
            )
            FastLanguageModel.for_inference(unsloth_model)
            self.model = unsloth_model
            if hasattr(unsloth_tokenizer, "tokenizer"):
                unsloth_tokenizer = unsloth_tokenizer.tokenizer
            self.tokenizer = unsloth_tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_name,
                device_map=cfg.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **quantization_kwargs,
            )
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

    def format_prompt(self, user_prompt: str, history: Optional[list[dict]] = None) -> str:
        """Wraps a raw clinical vignette prompt with the stream system prompt,
        using the tokenizer's chat template if available.

        `history` is an optional list of past turns, each
        `{"role": "user"|"assistant", "content": str}`, enabling multi-turn
        "chatbot" use (a clinician or patient asking follow-up questions).
        Past assistant turns should contain the FULL raw generation
        (including all four stream tags), not a cleaned/summarized version --
        this gives the model explicit access to its own past reasoning trace
        on follow-up turns, which is also directly useful for the
        R_retention reward (see core/rewards/retention.py).

        This tries up to four message-format variants, from most-specific
        requirement to most-general, and uses whichever the tokenizer accepts:
          1. system+user turns, STRUCTURED content (Gemma 3 / MedGemma
             requirement -- confirmed in Ahmed's own prior working repo:
             Gemma 3's chat template requires content as
             `[{"type": "text", "text": ...}]`, not a plain string, even for
             pure text input, and separately raises `TemplateError: System
             role not supported` if a system turn is included at all).
          2. system+user turns, PLAIN STRING content (most other backbones).
          3. merged-user-turn (system prompt folded into first user turn,
             Google's documented Gemma workaround), STRUCTURED content.
          4. merged-user-turn, PLAIN STRING content.
          5. Final fallback: plain-text prompt, no chat template at all.
        This lets the same code path work across Nemotron, MedGemma/Gemma 3,
        OctoMed, and GPT-OSS without per-backbone branching at the call site.
        """
        history = history or []

        def structured(role: str, content: str) -> dict:
            return {"role": role, "content": [{"type": "text", "text": content}]}

        def plain(role: str, content: str) -> dict:
            return {"role": role, "content": content}

        def try_apply(messages: list[dict]) -> Optional[str]:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return None

        # Attempt 1 & 2: system+user, structured then plain content
        for content_fn, label in [(structured, "structured"), (plain, "plain-string")]:
            try:
                messages = [content_fn("system", STREAM_SYSTEM_PROMPT)]
                messages.extend(history)
                messages.append(content_fn("user", user_prompt))
                result = try_apply(messages)
                if result is not None:
                    return result
            except Exception as e:
                logger.info(
                    "Tokenizer rejected system-role + %s content (%s) -- trying next format variant.",
                    label, type(e).__name__,
                )

        # Attempt 3 & 4: merged-user-turn (Gemma system-role workaround), structured then plain
        for content_fn, label in [(structured, "structured"), (plain, "plain-string")]:
            try:
                if history:
                    merged_history = [dict(h) for h in history]
                    first_user_idx = next((i for i, h in enumerate(merged_history) if h["role"] == "user"), None)
                    merged_text = f"{STREAM_SYSTEM_PROMPT}\n\n{user_prompt}"
                    if first_user_idx is not None:
                        # keep existing history content, just prepend the system prompt
                        # to the FIRST turn (not the new question) so instructions
                        # aren't repeated every follow-up turn
                        first_content = merged_history[first_user_idx]["content"]
                        first_text = first_content[0]["text"] if isinstance(first_content, list) else first_content
                        merged_history[first_user_idx] = content_fn("user", f"{STREAM_SYSTEM_PROMPT}\n\n{first_text}")
                        messages = merged_history + [content_fn("user", user_prompt)]
                    else:
                        messages = [content_fn("user", merged_text)] + merged_history
                else:
                    messages = [content_fn("user", f"{STREAM_SYSTEM_PROMPT}\n\n{user_prompt}")]
                result = try_apply(messages)
                if result is not None:
                    return result
            except Exception as e:
                logger.info(
                    "Tokenizer rejected merged-user-turn + %s content (%s) -- trying next format variant.",
                    label, type(e).__name__,
                )

        logger.warning(
            "All chat-template format variants failed or no chat template is available -- "
            "falling back to a plain-text prompt with no chat template applied at all. "
            "Verify this is acceptable for your chosen backbone."
        )

        # Final fallback if the backbone has no usable chat template at all
        history_text = "\n\n".join(f"### {h['role'].capitalize()}\n{h['content']}" for h in history)
        prefix = f"{history_text}\n\n" if history_text else ""
        return f"{STREAM_SYSTEM_PROMPT}\n\n{prefix}### Input\n{user_prompt}\n\n### Response\n"

    @torch.no_grad()
    def _generate_once(self, inputs, max_new_tokens: int, temperature: float, top_p: float,
                        num_return_sequences: int) -> list[str]:
        stopping_criteria = StoppingCriteriaList([_StopOnClosingTag(self.tokenizer)])
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def generate(self, user_prompt: str, history: Optional[list[dict]] = None, max_new_tokens: int = 768,
                 temperature: float = 0.7, top_p: float = 0.9,
                 num_return_sequences: int = 1, max_retries_if_malformed: int = 2) -> list[str]:
        """`max_retries_if_malformed`: if the (single) generation isn't a
        well-formed four-stream response, regenerate up to this many more
        times before giving up and returning the last attempt as-is.

        Added as a practical mitigation for a real, confirmed generation-
        termination defect (see _StopOnClosingTag's docstring and the
        README bug log): some checkpoints sometimes never converge to a
        complete, well-formed response at all, regardless of decoding
        config -- a problem the stopping criterion alone can't fix, since
        it only helps once a well-formed response HAS been produced. This
        doesn't fix the underlying cause; it trades a bounded amount of
        extra latency for a much lower chance of surfacing a malformed
        response to a caller, which is a reasonable trade for interactive
        use (Chat tab, evaluation) but is deliberately NOT applied when
        num_return_sequences > 1 (GRPO-style group sampling), where
        retrying only some sequences in a group would complicate the
        group-relative-advantage math in ways that need separate handling
        -- see training/grpo.py, which doesn't call this method at
        all (TRL's GRPOTrainer generates rollouts internally).
        """
        formatted = self.format_prompt(user_prompt, history=history)
        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                                 max_length=self.cfg.max_seq_len).to(self.model.device)

        if num_return_sequences != 1:
            return self._generate_once(inputs, max_new_tokens, temperature, top_p, num_return_sequences)

        attempts = max(1, max_retries_if_malformed + 1)
        last_result = None
        for attempt in range(attempts):
            result = self._generate_once(inputs, max_new_tokens, temperature, top_p, 1)
            last_result = result
            if parse_streams(result[0]).well_formed:
                if attempt > 0:
                    logger.info("Generation succeeded on retry %d/%d after an earlier malformed attempt.",
                                attempt + 1, attempts)
                return result
        logger.warning("Generation did not produce a well-formed response after %d attempt(s); "
                        "returning the last (malformed) attempt.", attempts)
        return last_result


class ConversationSession:
    """Manages a running multi-turn conversation ("chatbot" use case) against
    a MultiStreamModel. Each turn produces the full four-stream output;
    `ask()` returns the parsed streams for convenience while internally
    keeping the FULL raw generation in history (see format_prompt's
    docstring for why raw, not cleaned, history is used).

    Usage:
        session = ConversationSession(ts_model)
        parsed1 = session.ask("What is this patient's current risk?")
        parsed2 = session.ask("Why do you say that specifically?")  # follow-up, has context of turn 1
    """

    def __init__(self, model: MultiStreamModel, max_new_tokens: int = 768,
                 temperature: float = 0.7, top_p: float = 0.9):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history: list[dict] = []

    def ask(self, user_prompt: str) -> ParsedStreams:
        generations = self.model.generate(
            user_prompt, history=self.history, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, top_p=self.top_p, num_return_sequences=1,
        )
        raw_response = generations[0]
        self.history.append({"role": "user", "content": user_prompt})
        self.history.append({"role": "assistant", "content": raw_response})
        return parse_streams(raw_response)

    def reset(self):
        self.history = []



if __name__ == "__main__":
    # Smoke test of the parser only (no model load required)
    example = (
        "<think>HR trending up, MAP trending down -- possible early shock.</think>"
        "<patient_state>Tachycardic, borderline hypotensive, lactate pending.</patient_state>"
        "<forecast>MAP_6h: 58 [52-64]\nlactate_6h: 3.1 [2.4-3.9]</forecast>"
        "<user_belief>Reader is likely a bedside nurse; keep it concrete and action-oriented.</user_belief>"
    )
    parsed = parse_streams(example)
    print(parsed)
    assert parsed.well_formed
    assert "MAP_6h" in parsed.forecast_values
    assert parsed.forecast_values["MAP_6h"].value == 58.0

    # Also verify the "not applicable" opt-out path works
    example_na = (
        "<think>General question, no forecast needed.</think>"
        "<patient_state>N/A</patient_state>"
        "<forecast>not applicable</forecast>"
        "<user_belief>Clinician.</user_belief>"
    )
    parsed_na = parse_streams(example_na)
    assert parsed_na.well_formed
    assert parsed_na.forecast_values == {}
    print("Parser smoke test passed.")
