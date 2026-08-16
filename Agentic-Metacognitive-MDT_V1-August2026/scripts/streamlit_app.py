"""
scripts/streamlit_app.py

Interactive test harness for all four phases of this project, not just a
chat demo. Four tabs:

1. Chat / Generation -- talk to a real trained checkpoint (Phase 1 or
   Phase 2) and see its four-stream output broken out.
2. Hypergraph Safety Check (Phase 3) -- type a claimed patient-state and
   see InterimRuleBasedChecker / LearnedHypergraphChecker's score live,
   plus browse the mined hyperedges on disk.
3. Agentic Tool Use (Phase 4) -- paste or generate a <think> block, extract
   its tool calls, and execute them live against a real registry.
4. Reward Inspector -- paste any four-stream generation and a reference
   patient state, and see every one of GRPO's 9 reward components broken
   out individually, not just the aggregate the trainer logs.

CHANGES FROM THE PRIOR (chat-only) VERSION:
- MODEL_REGISTRY previously pointed at Ahmed's prior repository's Hub model
  IDs ("medgemma-4b-digital-twin-v1", etc.), none of which are this
  project's actual trained checkpoints -- selecting any of them here would
  have failed to load anything genuinely testable. Now points at this
  project's REAL local checkpoint directories.
- Extended from a single chat view into the four tabs above, since testing
  "the phases" (Phase 3's safety checker, Phase 4's tool-use loop, and the
  reward functions Phase 2 actually optimizes) needs more than a chat box.

USAGE:
    streamlit run scripts/streamlit_app.py

REQUIREMENTS: streamlit, torch, unsloth, transformers, peft
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# FIXED 2026-08-13: imports below previously referenced the old flat module
# layout (models.multi_stream, hypergraph.verification, agents.tool_use,
# training.rewards), none of which exist after the 2026-08-12 restructure
# into core/ + training/ -- this file had never actually been run since
# then, so the breakage was undetected until a real headless functional
# test (tests/test_streamlit_app_logic.py) imported it directly.
from training.backbone import MultiStreamModel, MultiStreamConfig, parse_streams, STREAM_SYSTEM_PROMPT
from core.hypergraph.verification import InterimRuleBasedChecker, LearnedHypergraphChecker
from core.tools.dispatch import extract_tool_calls, make_default_registry
from core.rewards.composite import compute_total_reward, RewardWeights

# ---------------------------------------------------------------------------
# Multi-model registry -- REAL local checkpoint directories from this
# project (not the prior repo's Hub IDs). checkpoint-135, the
# phase2_grpo_resume run's checkpoints, and every COLLAPSED_DO_NOT_USE
# directory are deliberately EXCLUDED -- see README bug log items 19-20 and
# the third-collapse update for why those are known NaN-corrupted and must
# never be loaded, even for casual testing.
#
# UPDATED 2026-08-13: the phase2_grpo/checkpoint-130 and phase1_sft entries
# below point at paths that do NOT exist in THIS repo -- they were the
# source repo's old (pre-v2/v3/v4-investigation, pre-restructure)
# checkpoints, never copied here. Only checkpoints/phase1_sft_v2_reproduction
# exists in this repo as of this fix (see PROGRESS.md for the full
# checkpoint history). Entries below are kept but marked with their real
# status so selecting a missing one fails with a clear message (load_model's
# existing try/except -> st.error), not a silent/confusing crash. Add real
# entries here once Phase 2 (job 39279689 as of this writing) actually
# produces a checkpoint in this repo.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Phase 1 SFT v2-reproduction (current working baseline, 46.7% format compliance)":
        str(PROJECT_ROOT / "checkpoints" / "phase1_sft_v2_reproduction"),
    "Phase 2 GRPO checkpoint-130 [NOT YET IN THIS REPO -- see comment above]":
        str(PROJECT_ROOT / "checkpoints" / "phase2_grpo" / "checkpoint-130"),
    "Phase 2 GRPO v2repro_base [pending -- job 39279689, not yet complete]":
        str(PROJECT_ROOT / "checkpoints" / "phase2_grpo_v2repro_base"),
    "Base MedGemma-4B (untrained, for comparison)": "google/medgemma-4b-it",
}

st.set_page_config(page_title="Medical Digital Twin -- Phase Tester", page_icon="🫀", layout="wide")

st.sidebar.title("Configuration")
model_label = st.sidebar.selectbox(
    "Model (used by the Chat and Reward Inspector tabs)",
    options=list(MODEL_REGISTRY.keys()),
    help="Switch between real trained checkpoints or add your own in MODEL_REGISTRY at the top of this file.",
)
custom_path = st.sidebar.text_input(
    "Or enter a custom local path / HF Hub ID", value="",
    help="Overrides the dropdown above if non-empty.",
)
model_path = custom_path.strip() if custom_path.strip() else MODEL_REGISTRY[model_label]

max_new_tokens = st.sidebar.slider("Max new tokens", 128, 2048, 768, step=64)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)

if st.sidebar.button("🔄 Reload model / clear cache"):
    st.cache_resource.clear()
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tabs 2-4 (hypergraph, tool use, reward inspector) run on CPU-only logic "
    "and don't need a model loaded -- only the Chat tab (and generating a "
    "fresh completion in the Reward Inspector) load the selected checkpoint."
)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        cfg = MultiStreamConfig(base_model_name=path)
        return MultiStreamModel(cfg)
    except Exception as e:
        st.error(f"Error loading model '{path}': {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_interim_checker():
    return InterimRuleBasedChecker()


@st.cache_resource(show_spinner=False)
def load_learned_checker(hypergraph_path: str):
    try:
        return LearnedHypergraphChecker(hypergraph_path, require_reviewed=True)
    except ValueError as e:
        return str(e)  # not reviewed yet -- surfaced as a message, not a crash


st.title("🫀 Medical Digital Twin -- Phase Tester")

tab_chat, tab_hypergraph, tab_tools, tab_rewards = st.tabs(
    ["💬 Chat / Generation", "🕸️ Hypergraph Safety Check (Phase 3)",
     "🔧 Agentic Tool Use (Phase 4)", "🎯 Reward Inspector"]
)

# ---------------------------------------------------------------------------
# Tab 1: Chat / Generation
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown(
        f"**Active model:** `{model_path}`  \n"
        "This system separates internal reasoning, patient-state summary, a structured "
        "forecast, and audience-adapted framing into four distinct sections before answering."
    )

    with st.spinner(f"Loading {model_label}..."):
        ts_model = load_model(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []  # raw {"role", "content"} turns, fed to format_prompt

    # Example questions spanning the range this architecture is meant to
    # handle: general medical reasoning (no patient-state/forecast expected),
    # real ICU-vignette-style deterioration scenarios (all four streams
    # expected, matching scripts/validate_phase1_checkpoint.py's and
    # validate_phase2_checkpoint.py's TEST_PROMPTS), a patient-facing
    # question (tests recipient-adapted framing / R_emp), and one that
    # explicitly invites a tool call (tests R_tool / the Phase 4 tool-use
    # path if the model attempts one). Click one to run it immediately.
    EXAMPLE_QUESTIONS = [
        "A 45-year-old male presents with chest pain radiating to the left "
        "arm, diaphoresis, and nausea for the past 30 minutes. What is the "
        "most likely diagnosis and immediate management?",
        "A 68-year-old male, 18 hours post-op from a bowel resection, has a "
        "heart rate of 112, MAP of 61, and a lactate that rose from 1.4 to "
        "2.6 over the last 4 hours. Assess deterioration risk.",
        "I just got back my labs and my doctor said my lactate is high. "
        "What does that mean and should I be worried?",
        "A 45-year-old female with type 2 diabetes was admitted for DKA. "
        "Current glucose 210, pH 7.32, and she reports feeling 'much "
        "better' than admission. Summarize her current state for her "
        "family, who are anxious and have limited medical background.",
        "Explain the mechanism by which ACE inhibitors can cause a dry cough.",
        "Check whether this combination is physiologically plausible: "
        "tachycardic, hypotensive, and hyperlactatemic. What does the "
        "hypergraph say about this pattern?",
    ]
    with st.expander("💡 Example questions to try", expanded=not st.session_state.messages):
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"example_{i}"):
                st.session_state.pending_prompt = q

    _GENERIC_PATIENT_STATE_PLACEHOLDER = "not applicable for this general-reasoning example"

    def _extract_answer_content(parsed) -> tuple[str, str]:
        """Returns (substantive_content, source_label). <user_belief> is a
        reader-framing note (per models/stream_parsing.py's docstring:
        "model's estimate of the recipient's knowledge/emotional state"),
        NOT itself an answer -- treating it as the whole "Response" (the
        prior version of this function) meant a general-reasoning question
        with a correctly-N/A <patient_state> displayed NOTHING but that
        generic framing note as the final answer, even when the model's
        <think> block contained a fully correct, substantive answer.
        Confirmed via a real generation: asked a genuine clinical question,
        the displayed "Response" was just "Assume a clinician-level reader
        for this general reasoning task." with the real answer invisible,
        buried in Reasoning. This picks the actual substantive content from
        whichever stream has it.
        """
        if parsed.patient_state and _GENERIC_PATIENT_STATE_PLACEHOLDER not in parsed.patient_state.lower():
            return parsed.patient_state, "patient_state"
        if parsed.think:
            # training/sft_formatting.py's format_reasoning_example appends
            # "Answer: <the real answer>" to the end of <think> (fixed
            # alongside this UI change -- see that module for the matching
            # data-pipeline bug) -- prefer that explicit marker if present.
            if "\nAnswer:" in parsed.think:
                return parsed.think.rsplit("\nAnswer:", 1)[1].strip(), "think (explicit answer)"
            # Fallback for checkpoints trained before that fix: no explicit
            # marker exists, so the closing sentence of the reasoning trace
            # is the best available approximation of a stated conclusion.
            sentences = [s.strip() for s in parsed.think.replace("\n", " ").split(".") if s.strip()]
            if sentences:
                return sentences[-1] + ".", "think (inferred conclusion, no explicit answer -- retrain with the sft_formatting.py fix for a real one)"
        return "", "none"

    def render_assistant_turn(parsed, raw_response: str):
        """Renders all four streams beside the final answer, always visible
        (no click-to-expand) -- both for a fresh generation and when
        re-displaying a past turn from history, so scrolling back up doesn't
        lose the reasoning/patient-state/forecast breakdown down to just the
        final answer text.
        """
        if not parsed.well_formed:
            st.warning(
                "Response did not match the expected four-stream format "
                "(this checkpoint may predate the forecast stream, or "
                "may not be well-formed for this generation). Showing raw output."
            )
            st.markdown(raw_response)
            return raw_response

        answer_col, streams_col = st.columns([3, 2])
        with answer_col:
            content, source = _extract_answer_content(parsed)
            st.markdown("**💬 Response**")
            if content:
                st.markdown(content)
            else:
                st.markdown("*(No substantive content found in any stream for this response.)*")
            if parsed.user_belief:
                st.caption(f"🎯 Framed for: {parsed.user_belief}")
            if source.startswith("think (inferred"):
                st.caption(f"⚠️ {source}")
            final_answer = content or parsed.user_belief or raw_response
        with streams_col:
            st.markdown("**🧠 Reasoning**")
            st.caption(parsed.think or "(empty)")
            st.markdown("**📋 Patient State**")
            st.caption(parsed.patient_state or "(empty)")
            st.markdown("**📈 Forecast**")
            if parsed.forecast_values:
                for var, entry in parsed.forecast_values.items():
                    st.caption(f"{var}: {entry.value} [{entry.low} - {entry.high}]")
            elif parsed.forecast and parsed.forecast.strip().lower() != "not applicable":
                st.caption(parsed.forecast)  # non-empty but didn't parse into the structured sub-format
            else:
                st.caption("Not applicable for this response.")
        return final_answer

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("parsed") is not None:
                render_assistant_turn(message["parsed"], message["raw"])
            else:
                st.markdown(message["content"])

    st.caption(
        "💡 This is a real conversation -- type a follow-up below and the model keeps the "
        "full context (including its own past reasoning, not just what's displayed) from "
        "every earlier turn in this chat."
    )
    chat_input_prompt = st.chat_input("Describe symptoms, ask a clinical question, or follow up...")
    prompt = st.session_state.pop("pending_prompt", None) or chat_input_prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if ts_model is not None:
            with st.chat_message("assistant"):
                with st.spinner("Generating structured response..."):
                    generations = ts_model.generate(
                        prompt, history=st.session_state.history,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        num_return_sequences=1,
                    )
                    raw_response = generations[0]

                parsed = parse_streams(raw_response)
                display_text = render_assistant_turn(parsed, raw_response)

                st.session_state.messages.append({
                    "role": "assistant", "content": display_text,
                    "parsed": parsed if parsed.well_formed else None, "raw": raw_response,
                })
                st.session_state.history.append({"role": "user", "content": prompt})
                st.session_state.history.append({"role": "assistant", "content": raw_response})
        else:
            st.error("Model failed to load -- check the path in the sidebar and try 'Reload model'.")

# ---------------------------------------------------------------------------
# Tab 2: Hypergraph Safety Check (Phase 3)
# ---------------------------------------------------------------------------
with tab_hypergraph:
    st.markdown(
        "Type a claimed patient-state sentence and see the score "
        "`training/rewards.py`'s `R_bound` term would assign it -- the "
        "safety-critical check that a generated `<patient_state>` isn't "
        "claiming a physiologically implausible or ungrounded combination "
        "of abnormalities."
    )

    checker_kind = st.radio(
        "Checker", ["Interim rule-based (available from day one)",
                    "Learned hypergraph (requires a CLINICALLY_REVIEWED hypergraph JSON)"],
        horizontal=True,
    )

    default_text = "Patient is tachycardic and hypotensive, consistent with early shock."
    claim_text = st.text_area("Claimed patient state", value=default_text, height=100)

    if checker_kind.startswith("Interim"):
        checker = load_interim_checker()
        score = checker.check(claim_text)
        st.metric("R_bound score", f"{score:.2f}", help="1.0 = no implausible pair detected; penalized per violation.")
    else:
        hg_path = st.text_input("Hypergraph JSON path", value=str(PROJECT_ROOT / "hypergraph" / "derived_hypergraph.json"))
        if Path(hg_path).exists():
            checker = load_learned_checker(hg_path)
            if isinstance(checker, str):
                st.warning(f"Cannot use this hypergraph as a safety constraint yet: {checker}")
            else:
                score = checker.check(claim_text)
                st.metric("R_bound score", f"{score:.2f}",
                          help="1.0 = grounded in (or too sparse to check against) a reviewed hyperedge; "
                               "0.2 = claims >=2 simultaneous abnormalities matching no known hyperedge.")
        else:
            st.info(f"No hypergraph file found at `{hg_path}` -- run Phase 3 (slurm/phase3_hypergraph.sbatch) first.")

    st.markdown("---")
    st.subheader("Browse mined hyperedges")
    browse_path = st.text_input("Path to browse", value=str(PROJECT_ROOT / "hypergraph" / "derived_hypergraph.json"), key="browse_path")
    if Path(browse_path).exists():
        with open(browse_path) as f:
            hg_data = json.load(f)
        status = hg_data.get("status", "UNKNOWN")
        st.markdown(f"**Status:** `{status}`" + ("  ⚠️ not yet clinically reviewed -- candidates only" if status != "CLINICALLY_REVIEWED" else ""))
        hyperedges = hg_data.get("hyperedges", [])
        st.markdown(f"**{len(hyperedges)} candidate hyperedges**")
        if hyperedges:
            st.dataframe(
                [{"variables": " + ".join(h["variables"]), "support": h["support"],
                  "p_value": h["p_value"], "odds_ratio": h["odds_ratio"]} for h in hyperedges[:200]],
                width="stretch",
            )
    else:
        st.info(f"No file found at `{browse_path}`.")

# ---------------------------------------------------------------------------
# Tab 3: Agentic Tool Use (Phase 4)
# ---------------------------------------------------------------------------
with tab_tools:
    st.markdown(
        "Paste (or generate, in the Chat tab) a `<think>` block containing "
        "structured tool calls, and see them extracted and executed against "
        "a real `ToolRegistry` -- exactly what `agents/tool_use.py`'s "
        "`run_agentic_turn` does mid-generation during a Phase 4 rollout."
    )

    default_think = (
        'Let me check this combination against known patterns. '
        '{"tool": "query_hypergraph", "args": {"claimed_abnormalities": ["tachycardia", "hypotension"]}} '
        'That looks consistent with shock physiology. Let me also check recent labs. '
        '{"tool": "get_recent_labs", "args": {"variable": "lactate", "hours": 6}}'
    )
    think_text = st.text_area("Think block", value=default_think, height=120)

    if st.button("Extract and execute tool calls"):
        calls = extract_tool_calls(think_text)
        if not calls:
            st.info("No tool calls found in this text (expects a JSON object with a \"tool\" key).")
        else:
            checker = load_interim_checker()
            # No per-patient timeseries context wired up in this standalone
            # tester (that only exists inside a real rollout worker, keyed
            # to a specific hadm_id -- see agents/rollout_service.py) --
            # get_recent_labs will correctly report itself as not configured
            # rather than fabricating patient data.
            registry = make_default_registry(checker)
            for call in calls:
                result = registry.call(call.get("tool"), call.get("args", {}))
                st.markdown(f"**Tool:** `{call.get('tool')}`  **Args:** `{call.get('args', {})}`")
                if result.success:
                    st.json(result.result)
                else:
                    st.error(f"Failed: {result.error}")
                st.caption(f"Latency: {result.latency_ms:.1f}ms")
                st.markdown("---")

# ---------------------------------------------------------------------------
# Tab 4: Reward Inspector
# ---------------------------------------------------------------------------
with tab_rewards:
    st.markdown(
        "Paste a full four-stream generation (or generate a fresh one below "
        "against the sidebar's selected checkpoint) and see every one of "
        "GRPO's 11 reward components scored individually -- what the trainer "
        "aggregates into a single `reward` number in its logs, broken back apart."
    )

    col1, col2 = st.columns(2)
    with col1:
        inspector_prompt = st.text_area(
            "Prompt (only used if you click Generate)",
            value="A 68-year-old male, 18 hours post-op, HR 112, MAP 61, lactate rising 1.4->2.6 over 4h.",
            height=80,
        )
        if st.button("Generate a completion with the selected model"):
            with st.spinner("Generating..."):
                model_for_reward = load_model(model_path)
            if model_for_reward is not None:
                st.session_state["inspector_generation"] = model_for_reward.generate(
                    inspector_prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                )[0]
    with col2:
        reference_patient_state = st.text_area(
            "Reference patient state (for R_sem, R_diagnostic)",
            value="Tachycardic and hypotensive with rising lactate, consistent with early septic/hypovolemic shock.",
            height=80,
        )
        recipient_type = st.selectbox("Recipient type (for R_emp)", ["clinician", "patient", "family"])

    col3, col4 = st.columns(2)
    with col3:
        # WITHOUT this field, R_forecast was silently unscoreable: with no
        # true_future_values, reward_forecast_accuracy (training/rewards.py)
        # treats "<forecast>not applicable</forecast>" as a PERFECT 1.0 and a
        # real numeric forecast as a MILDLY PENALIZED 0.5 -- the opposite of
        # what a case like the deterioration vignette below actually calls
        # for. Leave this blank to test the "no forecast expected" case
        # (e.g. a pure general-reasoning question); fill it in to test
        # whether the model correctly commits to a forecast when it should.
        future_values_text = st.text_input(
            "Expected future values (for R_forecast) -- e.g. MAP_6h=55, lactate_6h=3.4",
            value="MAP_6h=55, lactate_6h=3.4",
            help="Comma-separated VAR=value pairs matching the <forecast> stream's variable "
                 "naming (e.g. MAP_6h, lactate_6h). Leave blank if no forecast is expected for "
                 "this prompt -- opting out then correctly scores 1.0 instead of being tested "
                 "against values that were never actually expected.",
        )
    with col4:
        must_mention_text = st.text_input(
            "Must-mention facts (for R_retention), semicolon-separated",
            value="",
            help="e.g. 'documented penicillin allergy; prior ICU stay this admission' -- leave "
                 "blank if this generation isn't a multi-turn/long-context case with facts that "
                 "need to be retained (R_retention then correctly scores 1.0, not tested).",
        )

    # R_tom (added 2026-08-13, see core/rewards/theory_of_mind.py): without
    # these two fields, compute_total_reward's recipient_knows/
    # recipient_does_not_know default to None, and R_tom silently scores a
    # neutral 1.0 for every generation -- correct behavior for "no ground
    # truth available", but that means this tab could never actually be
    # used to inspect R_tom's real scoring unless these are wired in too.
    col5, col6 = st.columns(2)
    with col5:
        recipient_knows_text = st.text_input(
            "Recipient already knows (for R_tom), semicolon-separated",
            value="",
            help="e.g. 'the diagnosis or working clinical assessment' -- matches the fact "
                 "strings data/synthetic/vignette_generation.py and "
                 "core/cohort/grpo_prompts.py sample from. Leave blank for 'no ground truth' "
                 "(R_tom then correctly scores a neutral 1.0, not tested).",
        )
    with col6:
        recipient_does_not_know_text = st.text_input(
            "Recipient does NOT know (for R_tom), semicolon-separated",
            value="",
            help="Same fact-string format as the field to the left.",
        )

    def _parse_future_values(text: str) -> dict:
        result = {}
        for pair in text.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            var, _, val = pair.partition("=")
            try:
                result[var.strip()] = float(val.strip())
            except ValueError:
                st.warning(f"Could not parse '{pair}' as VAR=number -- skipped.")
        return result

    generation_text = st.text_area(
        "Generation to score (edit freely, or click Generate above to fill this in)",
        value=st.session_state.get("inspector_generation", ""),
        height=200,
    )

    if st.button("Score this generation") and generation_text.strip():
        checker = load_interim_checker()
        true_future_values = _parse_future_values(future_values_text)
        must_mention_facts = [f.strip() for f in must_mention_text.split(";") if f.strip()]
        recipient_knows = [f.strip() for f in recipient_knows_text.split(";") if f.strip()] or None
        recipient_does_not_know = [f.strip() for f in recipient_does_not_know_text.split(";") if f.strip()] or None
        reward_dict = compute_total_reward(
            generated_text=generation_text,
            reference_patient_state=reference_patient_state,
            hypergraph_checker=checker,
            recipient_type=recipient_type,
            must_mention_facts=must_mention_facts,
            true_future_values=true_future_values,
            recipient_knows=recipient_knows,
            recipient_does_not_know=recipient_does_not_know,
            weights=RewardWeights(),
        )
        if true_future_values:
            st.caption(f"Scoring R_forecast against: {true_future_values}")
        cols = st.columns(3)
        for i, (name, value) in enumerate(reward_dict.items()):
            with cols[i % 3]:
                st.metric(name, f"{value:.3f}" if isinstance(value, float) else str(value))
