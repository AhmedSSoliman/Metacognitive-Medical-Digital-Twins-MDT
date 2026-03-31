"""
Gradio web interface for Medical Digital Twin.
"""

import logging
from typing import Optional

import torch

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from core.cognitive_streams import CognitiveStreamParser
from models.mdt_model import MedicalDigitalTwinModel
from rewards.composite_engine import CompositeRewardEngine

logger = logging.getLogger(__name__)


def create_gradio_interface(
    model: MedicalDigitalTwinModel,
    parser: CognitiveStreamParser,
    reward_engine: Optional[CompositeRewardEngine] = None
):
    """Create Gradio web interface."""
    
    if not GRADIO_AVAILABLE:
        logger.error("Gradio not available. Install with: pip install gradio")
        return None

    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required for Gradio launch, but no GPU is available.")
        return None
    
    SYSTEM_PROMPT = """You are a Metacognitive Medical Digital Twin.

Your response MUST follow this structure:

<think>
1. Clinical Assessment
2. Differential Diagnosis
3. Risk Stratification
4. Self-Correction if needed
</think>

<patient_state>
- Relevant vitals and labs with LOINC codes
- Physiological trajectory
</patient_state>

<user_belief>
- Inferred literacy level
- Emotional state
- Communication strategy
</user_belief>

Then provide a clear, empathetic response.
"""

    def _content_to_text(content) -> str:
        """Normalize Gradio message content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(content)

    def _clean_model_output(raw_text: str, user_message: str) -> str:
        """Clean model output to avoid role/prompt echo artifacts."""
        text = (raw_text or "").strip()
        if not text:
            return ""

        # Keep assistant span if role headers are present
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1].strip()

        # Truncate if model starts generating the next turn role header
        stop_markers = [
            "\nUser:", "\nHuman:", "\nSystem:",
            "<|im_start|>user", "<|start_header_id|>user", "### User"
        ]
        cut_positions = [text.find(m) for m in stop_markers if m in text]
        if cut_positions:
            text = text[:min(cut_positions)].strip()

        # Remove direct question echo at the start
        user_clean = (user_message or "").strip()
        if user_clean and text.lower().startswith(user_clean.lower()):
            text = text[len(user_clean):].lstrip("\n :-")

        # If output is still only a repeated question, return empty so fallback can trigger
        if user_clean and text.strip().lower() == user_clean.lower():
            return ""

        return text.strip()

    def _normalize_history(history: Optional[list]) -> list[dict]:
        """Normalize incoming history to Gradio messages format."""
        normalized = []
        if not history:
            return normalized

        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                normalized.append({
                    "role": item["role"],
                    "content": _content_to_text(item.get("content", ""))
                })
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                normalized.append({"role": "user", "content": _content_to_text(user_msg)})
                normalized.append({"role": "assistant", "content": _content_to_text(assistant_msg)})

        return normalized
    
    def format_response_markdown(text: str) -> str:
        """Format response with collapsible sections."""
        streams = parser.parse(text)
        
        formatted = ""
        
        if streams.think:
            formatted += f"""
<details>
<summary><b>🧠 Clinical Reasoning</b></summary>
```
{streams.think}
```

</details>

"""
        
        if streams.patient_state:
            formatted += f"""
<details>
<summary><b>📊 Patient State Analysis</b></summary>
```
{streams.patient_state}
```

</details>

"""
        
        if streams.user_belief:
            formatted += f"""
<details>
<summary><b>🎯 Communication Strategy</b></summary>
```
{streams.user_belief}
```

</details>

"""
        
        # Extract main response
        main_response = text
        for tag in ['<think>', '</think>', '<patient_state>', '</patient_state>',
                    '<user_belief>', '</user_belief>']:
            main_response = main_response.replace(tag, '')
        
        formatted += "\n---\n\n### Clinical Response:\n\n" + main_response.strip()
        
        return formatted
    
    def generate_response(message: str, history: list) -> str:
        """Generate response."""
        try:
            conversation = SYSTEM_PROMPT + "\n\n"

            for msg in _normalize_history(history):
                if msg["role"] == "user":
                    conversation += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    conversation += f"Assistant: {msg['content']}\n\n"
            
            conversation += f"User: {message}\n\nAssistant:"
            
            response = model.generate(
                conversation,
                max_length=1024,
                temperature=0.7,
                top_p=0.9
            )

            response = _clean_model_output(response, message)
            if not response:
                response = "I understand your question. Let me provide a focused clinical response."
            
            formatted = format_response_markdown(response)
            
            return formatted
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # Example queries
    examples = [
        "I have chest pain radiating to my left arm. What should I do?",
        "My 5-year-old has a fever of 102°F. Should I be concerned?",
        "Can you explain my lab results? Lactate 3.2, Creatinine 1.9",
        "I'm on metformin but experiencing stomach issues. Options?"
    ]
    
    # Create interface
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="indigo"),
        title="Cognitive Medical Digital Twin"
    ) as interface:
        
        gr.Markdown(
            """
            # 🫀 Metacognitive Medical Digital Twin
            
            **Advanced Clinical AI with Triple-Stream Reasoning**
            
            - 🧠 Transparent Clinical Reasoning
            - 📊 Physiological Trajectory Prediction
            - 🎯 Empathetic Communication
            - ⚡ Early Warning Detection
            - 🛡️ Safety Constraints
            
            ⚠️ **Disclaimer:** Research prototype. Consult healthcare professionals.
            """
        )
        
        chatbot = gr.Chatbot(label="Clinical Consultation", height=500)
        
        msg_input = gr.Textbox(
            label="Your Question",
            placeholder="Describe your clinical question...",
            lines=3
        )
        
        with gr.Row():
            submit_btn = gr.Button("🔬 Consult MDT", variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        gr.Examples(examples=examples, inputs=msg_input)
        
        with gr.Accordion("ℹ️ System Info", open=False):
            gr.Markdown(
                f"""
                **Model:** {model.config.model_name}  
                **Device:** {model.device}  
                
                **Architecture:**
                - Triple-Stream Cognitive Reasoning
                - Theory of Mind Inference
                - Multi-Reward GRPO Alignment
                """
            )
        
        # Event handlers
        def respond(message, chat_history):
            messages = _normalize_history(chat_history)
            messages.append({"role": "user", "content": message})
            messages.append({"role": "assistant", "content": ""})

            conversation = SYSTEM_PROMPT + "\n\n"
            for msg in messages[:-1]:
                if msg["role"] == "user":
                    conversation += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    conversation += f"Assistant: {msg['content']}\n\n"

            conversation += f"User: {message}\n\nAssistant:"

            partial_response = ""
            try:
                # Add a block cursor while streaming to indicate thinking
                for partial_response in model.generate_stream(
                    conversation,
                    max_length=1024,
                    temperature=0.7,
                    top_p=0.9
                ):
                    raw_text = _clean_model_output(partial_response, message)
                    messages[-1]["content"] = raw_text + " ▌"
                    yield "", messages

                # Final formatting once generation is fully complete
                final_text = _clean_model_output(partial_response, message)
                if not final_text:
                    final_text = "I understand your question. Let me provide a focused clinical response."
                formatted_response = format_response_markdown(final_text)
                messages[-1]["content"] = formatted_response
                yield "", messages

            except Exception as e:
                messages[-1]["content"] = f"❌ Error: {str(e)}"
                yield "", messages
        
        msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
        submit_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
    
    return interface


def launch_web_interface():
    """Launch the Gradio web interface."""
    from config.configs import ModelConfig, CognitiveArchitectureConfig, GRPOConfig
    from models.mdt_model import MedicalDigitalTwinModel
    from core.cognitive_streams import CognitiveStreamParser
    from rewards.composite_engine import CompositeRewardEngine
    
    logger.info("Initializing Medical Digital Twin...")

    if not torch.cuda.is_available():
        logger.error("Cannot launch Gradio UI on CPU-only runtime. Please run on a CUDA-enabled GPU node.")
        return
    
    # Initialize components
    model_config = ModelConfig()
    cog_config = CognitiveArchitectureConfig()
    grpo_config = GRPOConfig()
    
    model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
    parser = CognitiveStreamParser(cog_config)
    engine = CompositeRewardEngine(
        w_semantic=grpo_config.w_semantic,
        w_metacognitive=grpo_config.w_metacognitive,
        w_empathy=grpo_config.w_empathy,
        w_proactivity=grpo_config.w_proactivity,
        w_safety=grpo_config.w_safety
    )
    
    # Create interface
    interface = create_gradio_interface(model, parser, engine)
    
    if interface:
        logger.info("Launching web interface...")
        interface.launch(share=True)
    else:
        logger.error("Failed to create interface")