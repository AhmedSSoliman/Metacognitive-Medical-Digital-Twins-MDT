"""
Medical Digital Twin model wrapper.

Handles model loading, LoRA configuration, and generation.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from config.configs import ModelConfig

logger = logging.getLogger(__name__)


class MedicalDigitalTwinModel:
    """Medical Digital Twin model wrapper."""
    
    def __init__(self, config: ModelConfig, use_demo_model: bool = True):
        """Initialize model."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_demo_model:
            logger.warning("Using GPT-2 for demo. Set use_demo_model=False for production.")
            self._setup_demo_model()
        else:
            self._setup_production_model()
    
    def _setup_demo_model(self):
        """Setup GPT-2 demo model."""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        logger.info("Loading GPT-2 demo model...")
        
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<think>", "</think>",
                "<patient_state>", "</patient_state>",
                "<user_belief>", "</user_belief>"
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        logger.info("Demo model loaded successfully")
    
    def _setup_production_model(self):
        """Setup Model with LoRA."""
        if not PEFT_AVAILABLE:
            logger.error("PEFT not available, cannot load production model")
            raise ImportError("PEFT required for production model")
        
        logger.info(f"Loading {self.config.model_name}...")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.use_nested_quant
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<think>", "</think>",
                "<patient_state>", "</patient_state>",
                "<user_belief>", "</user_belief>"
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Production model loaded")
        self.model.print_trainable_parameters()
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        max_length = max_length or self.config.max_length
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                top_k=kwargs.get('top_k', self.config.top_k),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        **kwargs
    ):
        """Generate text from prompt as a stream."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        max_length = max_length or self.config.max_length
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=False
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_length=max_length,
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            top_k=kwargs.get('top_k', self.config.top_k),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        if PEFT_AVAILABLE:
            if isinstance(self.model, PeftModel):
                # Load adapter weights into existing PEFT model
                self.model.load_adapter(str(checkpoint_path), "default")
            else:
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        
        logger.info("Checkpoint loaded")