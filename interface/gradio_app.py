"""
Gradio web interface for Medical Digital Twin.
"""

import logging
from typing import Optional

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
            
            for user_msg, assistant_msg in history:
                conversation += f"User: {user_msg}\n"
                conversation += f"Assistant: {assistant_msg}\n\n"
            
            conversation += f"User: {message}\n\nAssistant:"
            
            response = model.generate(
                conversation,
                max_length=1024,
                temperature=0.7,
                top_p=0.9
            )
            
            response = response.split("Assistant:")[-1].strip()
            
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
            bot_message = generate_response(message, chat_history)
            chat_history.append([message, bot_message])
            return "", chat_history
        
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