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
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self._ensure_generation_readiness()
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
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._ensure_generation_readiness()
        logger.info("Production model loaded")
        self.model.print_trainable_parameters()

    def _ensure_generation_readiness(self) -> None:
        """Validate tokenizer/model consistency to avoid CUDA device-side asserts."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized before generation checks.")

        embedding_rows = int(self.model.get_input_embeddings().num_embeddings)
        tokenizer_size = int(len(self.tokenizer))

        # Try to align embedding matrix with tokenizer size when possible.
        if embedding_rows != tokenizer_size:
            logger.warning(
                "Tokenizer/model size mismatch detected (tokenizer=%s, embeddings=%s). "
                "Attempting resize_token_embeddings.",
                tokenizer_size,
                embedding_rows,
            )
            self.model.resize_token_embeddings(tokenizer_size)
            embedding_rows = int(self.model.get_input_embeddings().num_embeddings)

        if embedding_rows != tokenizer_size:
            raise ValueError(
                "Tokenizer/model vocab mismatch after resize attempt: "
                f"tokenizer={tokenizer_size}, embeddings={embedding_rows}."
            )

        # Ensure special token ids are valid.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer is missing both pad_token_id and eos_token_id.")

        for name, token_id in {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }.items():
            if token_id is None:
                continue
            if int(token_id) < 0 or int(token_id) >= embedding_rows:
                raise ValueError(
                    f"Invalid {name}={token_id} for embedding rows={embedding_rows}."
                )

        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            if hasattr(output_embeddings, "out_features"):
                output_rows = int(output_embeddings.out_features)
            elif hasattr(output_embeddings, "weight"):
                output_rows = int(output_embeddings.weight.shape[0])
            else:
                output_rows = None

            if output_rows is not None and output_rows != embedding_rows:
                if hasattr(self.model, "tie_weights"):
                    logger.warning(
                        "Input/output embedding mismatch detected (input_rows=%s, output_rows=%s). "
                        "Attempting tie_weights().",
                        embedding_rows,
                        output_rows,
                    )
                    self.model.tie_weights()
                    output_embeddings = self.model.get_output_embeddings()
                    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
                        output_rows = int(output_embeddings.weight.shape[0])
                    elif output_embeddings is not None and hasattr(output_embeddings, "out_features"):
                        output_rows = int(output_embeddings.out_features)

            if output_rows is not None and output_rows != embedding_rows:
                raise ValueError(
                    "Input/output embedding size mismatch detected: "
                    f"input_rows={embedding_rows}, output_rows={output_rows}."
                )

    def _get_prompt_debug_info(self, prompt: str, max_prompt_tokens: int) -> dict:
        """Build prompt tokenization diagnostics for crash triage."""
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = tokenized["input_ids"]

        embedding_rows = int(self.model.get_input_embeddings().num_embeddings)
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            output_rows = int(output_embeddings.weight.shape[0])
        elif output_embeddings is not None and hasattr(output_embeddings, "out_features"):
            output_rows = int(output_embeddings.out_features)
        else:
            output_rows = None

        return {
            "prompt_chars": len(prompt),
            "prompt_token_count": int(input_ids.shape[-1]),
            "prompt_token_min": int(input_ids.min().item()),
            "prompt_token_max": int(input_ids.max().item()),
            "tokenizer_size": int(len(self.tokenizer)),
            "embedding_rows": embedding_rows,
            "output_rows": output_rows,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        self._ensure_generation_readiness()
        requested_total_length = int(max_length or self.config.max_length)

        # Determine an effective context window from model/tokenizer constraints.
        model_max_positions = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)

        context_candidates = [requested_total_length]
        if isinstance(model_max_positions, int) and model_max_positions > 0:
            context_candidates.append(model_max_positions)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 1_000_000:
            context_candidates.append(tokenizer_max_length)

        effective_context = max(2, min(context_candidates))

        # Keep at least one token budget for generation.
        max_prompt_tokens = max(1, effective_context - 1)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens
        ).to(self.device)

        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[-1])
        available_new_tokens = max(1, effective_context - prompt_len)

        # Preserve prior API semantics where max_length was "total sequence length".
        requested_new_tokens = max(1, requested_total_length - prompt_len)
        max_new_tokens = min(requested_new_tokens, available_new_tokens)

        # Defensive checks: fail fast on invalid token ids before CUDA kernels run.
        embedding_rows = int(self.model.get_input_embeddings().num_embeddings)
        token_id_min = int(input_ids.min().item())
        token_id_max = int(input_ids.max().item())
        if token_id_min < 0 or token_id_max >= embedding_rows:
            raise ValueError(
                "Tokenizer/model vocab mismatch detected before generation: "
                f"token id range=({token_id_min}, {token_id_max}), "
                f"embedding rows={embedding_rows}."
            )

        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            if hasattr(output_embeddings, "weight"):
                output_rows = int(output_embeddings.weight.shape[0])
            elif hasattr(output_embeddings, "out_features"):
                output_rows = int(output_embeddings.out_features)
            else:
                output_rows = embedding_rows

            if token_id_max >= output_rows:
                raise ValueError(
                    "Tokenizer/model output vocab mismatch detected before generation: "
                    f"token id max={token_id_max}, output rows={output_rows}."
                )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
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
        self._ensure_generation_readiness()
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        requested_total_length = int(max_length or self.config.max_length)

        model_max_positions = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)

        context_candidates = [requested_total_length]
        if isinstance(model_max_positions, int) and model_max_positions > 0:
            context_candidates.append(model_max_positions)
        if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 1_000_000:
            context_candidates.append(tokenizer_max_length)

        effective_context = max(2, min(context_candidates))
        max_prompt_tokens = max(1, effective_context - 1)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens
        ).to(self.device)

        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[-1])
        available_new_tokens = max(1, effective_context - prompt_len)
        requested_new_tokens = max(1, requested_total_length - prompt_len)
        max_new_tokens = min(requested_new_tokens, available_new_tokens)

        embedding_rows = int(self.model.get_input_embeddings().num_embeddings)
        token_id_min = int(input_ids.min().item())
        token_id_max = int(input_ids.max().item())
        if token_id_min < 0 or token_id_max >= embedding_rows:
            raise ValueError(
                "Tokenizer/model vocab mismatch detected before streaming generation: "
                f"token id range=({token_id_min}, {token_id_max}), "
                f"embedding rows={embedding_rows}."
            )
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=False
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
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
        
        # Ensure tokenizer/model remain aligned after loading checkpoint artifacts.
        self._ensure_generation_readiness()
        logger.info("Checkpoint loaded")