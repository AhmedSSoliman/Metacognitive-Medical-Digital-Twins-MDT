"""
Configuration classes for Medical Digital Twin system.

Contains all hyperparameters, paths, and settings organized into
domain-specific dataclasses for clean separation of concerns.

Configuration Modules:
    - ModelConfig: Base model and LoRA fine-tuning parameters
    - CognitiveArchitectureConfig: Triple-stream XML markers and validation
    - DataConfig: Dataset paths, LOINC codes, preprocessing settings
    - SFTConfig: Supervised fine-tuning (Phase 1) hyperparameters
    - GRPOConfig: Group Relative Policy Optimization (Phase 4) settings
    - EvaluationConfig: Test cases and metrics configuration

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """
    Model architecture and LoRA fine-tuning configuration.
    
    Specifies base foundation model and parameter-efficient fine-tuning
    settings using Low-Rank Adaptation (LoRA) with 4-bit quantization.
    """
    
    # Base foundation model
    #model_name: str = "google/medgemma-4b-it"  # MedGemma-4B Instruction-Tuned
    #model_name: str = "unsloth/Qwen3.5-4B" # Qwen3.5-4B is a strong open-source alternative to MedGemma-4B
    model_name: str = "Qwen/Qwen3.5-4B" # Hugging Face version of Qwen3.5-4B
    max_length: int = 2048  # Maximum sequence length (tokens)    


    # LoRA (Low-Rank Adaptation) parameters
    lora_r: int = 16  # Rank of LoRA matrices (higher = more capacity, slower)
    lora_alpha: int = 32  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.05  # Dropout probability in LoRA layers
    
    # Target modules for LoRA adaptation
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",      # Query projection (attention)
        "v_proj",      # Value projection (attention)
        "k_proj",      # Key projection (attention)
        "o_proj",      # Output projection (attention)
        "gate_proj",   # Gating mechanism (FFN)
        "up_proj",     # Up-projection (FFN)
        "down_proj"    # Down-projection (FFN)
    ])
    
    # 4-bit quantization (QLoRA) for memory efficiency
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"  # Computation precision
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4 quantization
    use_nested_quant: bool = False  # Nested quantization for additional compression
    
    # Inference settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


@dataclass
class CognitiveArchitectureConfig:
    """
    Triple-stream cognitive architecture settings.
    
    Defines XML markers for the three auditable reasoning streams:
        - <think>: Deductive clinical logic and differential diagnosis
        - <patient_state>: Physiological state with LOINC-coded biomarkers
        - <user_belief>: Theory of Mind inference for empathetic communication
    """
    
    # Stream opening/closing markers
    think_marker: str = "<think>"
    think_end_marker: str = "</think>"
    
    patient_state_marker: str = "<patient_state>"
    patient_state_end_marker: str = "</patient_state>"
    
    user_belief_marker: str = "<user_belief>"
    user_belief_end_marker: str = "</user_belief>"
    
    # Validation requirements
    require_all_streams: bool = True  # All three streams must be present
    min_think_length: int = 50  # Minimum characters in <think> stream
    min_patient_state_length: int = 20  # Minimum characters in <patient_state>
    min_user_belief_length: int = 20  # Minimum characters in <user_belief>
    
    # Stream processing
    allow_nested_markers: bool = False  # Disallow nested XML tags
    strip_whitespace: bool = True  # Remove leading/trailing whitespace


@dataclass
class DataConfig:
    """
    Dataset paths and preprocessing parameters.
    
    Configures MIMIC-IV critical care data, Medical-O1 reasoning dataset,
    and standardized clinical ontology mappings (LOINC codes).
    """
    
    # MIMIC-IV Critical Care Database configuration
    mimic_root_dir: str = "mimiciv/3.1"  # Root directory for MIMIC-IV v3.1
    mimic_version: str = "3.1"  # Dataset version (important for schema changes)
    min_icu_stay_hours: int = 24  # Filter stays shorter than this duration
    
    # Medical-O1 Reasoning Dataset
    medical_o1_dataset: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    medical_o1_config: str = "en"  # Config: 'en', 'zh', 'en_mix', 'zh_mix'
    
    
    # LOINC (Logical Observation Identifiers Names and Codes) mappings
    # Maps biomarker names to standardized LOINC codes for EHR interoperability
    loinc_codes: Dict[str, str] = field(default_factory=lambda: {
        'heart_rate': '8867-4',           # Heart rate (beats/min)
        'sbp': '8480-6',                  # Systolic blood pressure (mmHg)
        'dbp': '8462-4',                  # Diastolic blood pressure (mmHg)
        'spo2': '59408-5',                # Oxygen saturation (%)
        'temperature': '8310-5',          # Body temperature (°C or °F)
        'respiratory_rate': '9279-1',     # Respiratory rate (breaths/min)
        'lactate': '2524-7',              # Serum lactate (mmol/L)
        'creatinine': '2160-0'            # Serum creatinine (mg/dL)
    })
    
    # Data preprocessing settings
    max_sequence_length: int = 2048  # Maximum tokens per training example
    train_test_split: float = 0.9  # 90% train, 10% test
    
    # MIMIC-IV specific processing
    outlier_removal: bool = True  # Remove physiologically impossible values
    forward_fill_hours: int = 2  # Interpolate missing values within this window
    temporal_alignment_window_minutes: int = 60  # Align measurements within 1 hour
    
    # Medical-O1 specific processing
    filter_incomplete_reasoning: bool = True  # Remove examples without full CoT
    max_o1_reasoning_length: int = 1500  # Maximum tokens for reasoning chains


@dataclass
class SFTConfig:
    """
    Supervised Fine-Tuning (Phase 1) configuration.
    
    Establishes triple-stream cognitive architecture through domain-specific
    supervised learning on MIMIC-IV + Medical-O1 datasets.
    
    Expected Training Time: 8-24 hours on A100 80GB GPU
    """
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4  # Per-device batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 4 * 8 = 32
    learning_rate: float = 2e-5  # Learning rate for AdamW optimizer
    warmup_ratio: float = 0.1  # 10% of steps for learning rate warmup
    
    # Optimization settings
    optimizer_type: str = "adamw_torch"  # Optimizer: 'adamw_torch', 'adamw_hf', 'sgd'
    weight_decay: float = 0.01  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    lr_scheduler_type: str = "cosine"  # Learning rate schedule
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Mixed precision training
    fp16: bool = False  # Use FP16 mixed precision (for V100)
    bf16: bool = True  # Use BF16 mixed precision (for A100)
    
    # Logging and checkpointing
    output_dir: str = "./outputs/sft"  # Model checkpoint directory
    logging_dir: str = "./outputs/logs/sft"  # TensorBoard logs
    logging_steps: int = 10  # Log every N steps
    save_steps: int = 500  # Save checkpoint every N steps
    save_total_limit: int = 3  # Keep only last 3 checkpoints
    
    # Evaluation during training
    eval_steps: int = 500  # Evaluate every N steps
    evaluation_strategy: str = "steps"  # Evaluate based on steps (not epochs)
    eval_accumulation_steps: int = 4  # Gradient accumulation for evaluation
    
    # Data loading
    dataloader_num_workers: int = 4  # Parallel data loading workers
    dataloader_pin_memory: bool = True  # Pin memory for faster GPU transfer
    
    # Reproducibility
    seed: int = 42


@dataclass
class GRPOConfig:
    """
    Group Relative Policy Optimization (Phase 4) configuration.
    
    Multi-objective reinforcement learning alignment balancing five rewards:
        - R_sem (0.25): Semantic fidelity (clinical accuracy)
        - R_meta (0.20): Metacognitive depth (self-correction)
        - R_emp (0.15): Structural empathy (communication calibration)
        - R_physio (0.25): Proactive surge prediction (early warning)
        - R_bound (0.15): Biological safety constraints
    
    Expected Training Time: 12-48 hours on A100 80GB GPU
    """
    
    # GRPO-specific hyperparameters
    num_iterations: int = 1000  # Total policy optimization iterations
    batch_size: int = 32  # Number of prompts per batch
    num_generations_per_prompt: int = 32  # K generations for group-relative advantages
    learning_rate: float = 1e-5  # Lower LR for fine-grained policy adjustment
    
    # Policy gradient settings
    clip_range: float = 0.2  # PPO-style clipping for stability
    gamma: float = 0.99  # Discount factor for multi-step rewards
    target_kl: float = 0.1  # Target KL divergence (early stopping)
    gae_lambda: float = 0.95  # Generalized Advantage Estimation lambda
    
    # Reward component weights (must sum to 1.0)
    w_semantic: float = 0.25  # Clinical accuracy (ROUGE-L + BERTScore)
    w_metacognitive: float = 0.20  # Self-correction depth (Delta-Embedding)
    w_empathy: float = 0.15  # Communication calibration (readability)
    w_proactivity: float = 0.25  # Early surge detection (anticipation)
    w_safety: float = 0.15  # Biological plausibility (range constraints)
    
    # Value function training
    vf_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    
    # Training settings
    num_train_epochs: int = 1  # Epochs per GRPO iteration
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Generation settings for policy rollouts
    generation_max_length: int = 1024  # Max tokens per generation
    generation_temperature: float = 0.8  # Sampling temperature
    generation_top_p: float = 0.9  # Nucleus sampling
    
    # Logging and checkpointing
    output_dir: str = "./outputs/grpo"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 5
    
    # Evaluation
    eval_every_n_iterations: int = 50  # Evaluate policy every N iterations
    
    # Reproducibility
    seed: int = 42


@dataclass
class EvaluationConfig:
    """
    Evaluation metrics and test case configuration.
    
    Defines comprehensive benchmarking across clinical scenarios and
    quantitative metrics for system validation.
    """
    
    # Test case configuration
    num_test_cases: int = 8  # Number of clinical scenarios to evaluate
    
    # Metric computation flags
    compute_bertscore: bool = True  # Semantic similarity (slow but accurate)
    compute_rouge: bool = True  # N-gram overlap metrics (fast)
    compute_bleu: bool = False  # BLEU typically not suitable for clinical text
    
    # Threshold settings for pass/fail
    min_keyword_coverage: float = 0.7  # 70% of expected keywords present
    min_stream_completeness: float = 0.8  # 80% of streams properly formatted
    min_safety_compliance: float = 1.0  # 100% compliance (no violations tolerated)
    min_self_correction_rate: float = 0.6  # 60% of cases show genuine revision
    
    # BERTScore settings
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  # High-quality model
    bertscore_device: str = "cuda"  # GPU acceleration
    
    # Stream validation settings
    validate_loinc_codes: bool = True  # Verify LOINC code format
    validate_physiological_ranges: bool = True  # Check biomarker plausibility
    validate_literacy_calibration: bool = True  # Ensure appropriate reading level
    
    # Output settings
    save_detailed_results: bool = True  # Save per-case breakdowns
    generate_visualizations: bool = True  # Create plots/charts
    output_format: str = "csv"  # Output format: 'csv', 'json', or 'both'


@dataclass
class HiPerGatorConfig:
    """
    HiPerGator HPC cluster-specific configuration.
    
    Settings for running on University of Florida's high-performance
    computing infrastructure. Optional - only used when detected.
    """
    
    # User configuration (update these)
    user_group: str = "your-group"  # SLURM account/group
    username: str = "your-username"  # HiPerGator username
    
    # File paths on HiPerGator
    mimic_root: str = f"/blue/{{user_group}}/{{username}}/mimic-iv-3.1"
    scratch_dir: str = f"/blue/{{user_group}}/{{username}}/scratch"
    output_dir: str = f"/blue/{{user_group}}/{{username}}/mdt-outputs"
    
    # SLURM resource allocation
    gpu_partition: str = "gpu"  # or "hpg-ai" for A100s
    num_gpus: int = 1
    gpu_type: str = "a100"  # "a100", "rtx6000", "v100"
    memory_gb: int = 64
    time_hours: int = 48
    
    # Module loading
    cuda_module: str = "cuda/11.8"
    conda_module: str = "conda"
    
    def get_mimic_config(self) -> DataConfig:
        """
        Generate DataConfig with HiPerGator-specific paths.
        
        Returns:
            DataConfig with paths pointing to Blue storage
        """
        return DataConfig(
            mimic_root_dir=self.mimic_root.format(
                user_group=self.user_group,
                username=self.username
            ),
            mimic_version="3.1"
        )