import os
#!/usr/bin/env python
# coding: utf-8

# # Metacognitive Medical Digital Twins Pipeline
# 
# This notebook aligns with the core principles of the Metacognitive Medical Digital Twins proposal. It demonstrates the instantiation, processing, and evaluation of patient digital twins.

# ## 1. HiPerGator Setup & Environment Activation
# 
# When running this on UF HiPerGator, you first need to load the Conda module and activate the environment. If you are not on HiPerGator, you can skip to the Python cells. 
# In a terminal, you would run the following:
# 
# ```bash
# # Load the ML environment on HiPerGator
# module load conda
# 
# # Activate the dedicated project environment
# conda activate digitaltwins_env
# ```
# 
# After activation, start your jupyter server/kernel with the same environment.

# # Medical Digital Twin - Master Notebook
# 
# **Complete End-to-End Training Pipeline**
# 
# This notebook runs the entire Medical Digital Twin system from start to finish:
# 1. Environment setup and dependency installation
# 2. Data preparation (MIMIC-IV + Medical-O1)
# 3. System testing and validation
# 4. Phase 1: Supervised Fine-Tuning (SFT)
# 5. Phase 4: Group Relative Policy Optimization (GRPO)
# 6. Evaluation and metrics
# 7. Web interface deployment
# 
# **Author:** Ahmed Soliman  
# **Institution:** University of Florida, HOBI  
# **Date:** 2026-03-21
# 
# ---

# ## **SECTION 1: Environment Setup**

# In[1]:


# Check Python version
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Should be Python 3.8+


# In[ ]:


# Install required packages
os.system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
os.system('pip install transformers datasets accelerate bitsandbytes')
os.system('pip install sentence-transformers pandas numpy tqdm')
os.system('pip install gradio matplotlib seaborn')
os.system('pip install peft unsloth')


# In[2]:


# Verify GPU availability
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected! Training will be very slow.")


# ## **SECTION 2: Import Project Modules**

# In[3]:


# Add project to path
import os
import sys
from pathlib import Path

# Assuming notebook is in project root
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")


# In[4]:


# Import project modules
from config.configs import (
    ModelConfig,
    CognitiveArchitectureConfig,
    DataConfig,
    SFTConfig,
    GRPOConfig,
    EvaluationConfig
)

from core.cognitive_streams import CognitiveStreamParser
from core.theory_of_mind import TheoryOfMindModule

from rewards.composite_engine import CompositeRewardEngine

from data.mimic_processor import MIMICProcessor
from data.medical_o1_processor import MedicalO1Processor
from data.dataset import CognitiveStreamDataset
from data.think_alignment import apply_soft_think_alignment

from models.mdt_model import MedicalDigitalTwinModel

from training.sft_trainer import run_sft_training
from training.grpo_trainer import run_grpo_training

from evaluation.evaluator import MedicalTwinEvaluator

from utils.helpers import clean_memory, setup_logging

print("✓ All modules imported successfully!")


# ## **SECTION 3: Configuration**

# In[5]:


# Setup logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("MEDICAL DIGITAL TWIN - MASTER PIPELINE")
logger.info("="*80)


# In[6]:


# Initialize configurations
model_config = ModelConfig()
cog_config = CognitiveArchitectureConfig()
data_config = DataConfig()
sft_config = SFTConfig()
grpo_config = GRPOConfig()
eval_config = EvaluationConfig()

# Training parameters (MODIFY THESE)
MAX_PATIENTS = 10  # Start small for testing (10), increase for production (1000)
MAX_O1_EXAMPLES = 100  # Medical-O1 examples to load
USE_DEMO_MODEL = True  # True=GPT-2 (fast), False=MedGemma-4B (slow but better)

print("Configuration:")
print(f"  Base model: {model_config.model_name}")
print(f"  Max patients: {MAX_PATIENTS}")
print(f"  Max O1 examples: {MAX_O1_EXAMPLES}")
print(f"  Demo mode: {USE_DEMO_MODEL}")
print(f"  Output directory: {sft_config.output_dir}")


# ## **SECTION 4: System Tests**

# In[7]:


# Run comprehensive system tests
print("\n" + "="*80)
print("RUNNING SYSTEM TESTS")
print("="*80 + "\n")

# Test 1: Cognitive Stream Parser
print("Test 1: Cognitive Stream Parser...")
parser = CognitiveStreamParser(cog_config)
test_text = """
<think>Patient presents with sepsis based on elevated lactate and clinical signs</think>
<patient_state>HR: 125 bpm [LOINC:8867-4], Lactate: 3.2 mmol/L [LOINC:2524-7]</patient_state>
<user_belief>Literacy: low, Emotional state: anxious</user_belief>
"""
streams = parser.parse(test_text)
is_valid, error = parser.validate(streams)
assert is_valid, f"Validation failed: {error}"
assert streams.is_complete(), "Streams incomplete"
print("✓ PASSED\n")

# Test 2: Theory of Mind
print("Test 2: Theory of Mind Module...")
tom = TheoryOfMindModule()
user_emb = torch.randn(1, 768)
with torch.no_grad():
    literacy, conf = tom.infer_literacy_level(user_emb)
print(f"✓ PASSED (literacy: {literacy.value}, confidence: {conf:.2f})\n")

# Test 3: Reward Engine
print("Test 3: Composite Reward Engine...")
reward_engine = CompositeRewardEngine()
print("✓ PASSED\n")

print("="*80)
print("ALL TESTS PASSED!")
print("="*80 + "\n")


# ## **SECTION 5: Data Preparation**

# In[8]:


# Check MIMIC-IV availability
print("\n" + "="*80)
print("DATA PREPARATION")
print("="*80 + "\n")

mimic_processor = MIMICProcessor(data_config)
mimic_available = mimic_processor.check_availability()

if mimic_available:
    print(f"✓ MIMIC-IV v{data_config.mimic_version} found at: {data_config.mimic_root_dir}")
else:
    print(f"❌ MIMIC-IV NOT FOUND at: {data_config.mimic_root_dir}")
    print("\nTo download MIMIC-IV:")
    print("1. Apply for access: https://physionet.org/content/mimiciv/3.1/")
    print("2. Complete CITI training")
    print("3. Download and extract to configured path")
    raise FileNotFoundError("MIMIC-IV data required to continue")


# In[9]:


# Process MIMIC-IV data
print(f"\nProcessing MIMIC-IV data (up to {MAX_PATIENTS} patients)...")

mimic_data = mimic_processor.process_all_patients(
    max_patients=MAX_PATIENTS,
    save_path="./outputs/mimic_processed.json"
)
mimic_data = [{**item, 'source': 'mimic'} for item in mimic_data]

print(f"✓ Processed {len(mimic_data)} MIMIC examples")
print(f"  Saved to: ./outputs/mimic_processed.json")


# In[12]:


# Load Medical-O1 data
print(f"\nLoading Medical-O1 reasoning dataset...")

try:
    o1_processor = MedicalO1Processor()
    o1_dataset = o1_processor.load_dataset('train', config=data_config.medical_o1_config)

    if o1_dataset:
        o1_data = o1_processor.format_for_training(
            o1_dataset,
            max_examples=MAX_O1_EXAMPLES
        )
        o1_data = [{**item, 'source': 'medical_o1'} for item in o1_data]
        print(f"✓ Loaded {len(o1_data)} Medical-O1 examples")
    else:
        print("⚠️ Medical-O1 dataset not available, using MIMIC only")
        o1_data = []
except Exception as e:
    print(f"⚠️ Medical-O1 loading failed: {e}")
    o1_data = []


# In[11]:


# Combine datasets
all_training_data = mimic_data + o1_data

if data_config.soft_think_enabled:
    all_training_data, alignment_summary = apply_soft_think_alignment(
        all_training_data,
        config=data_config
    )
    print("\n✓ Applied soft-mandatory CoT alignment")
    print(f"  Gold think samples: {alignment_summary['gold_examples']}")
    print(f"  Synthetic think samples: {alignment_summary['synthetic_examples']}")
    print(f"  Synthetic quality pass rate: {alignment_summary['quality_pass_rate']:.2%}")
    print(f"  Average think weight: {alignment_summary['avg_think_weight']:.3f}")

print(f"\n✓ Total training examples: {len(all_training_data)}")
print(f"  MIMIC-IV: {len(mimic_data)}")
print(f"  Medical-O1: {len(o1_data)}")


# ## **SECTION 6: Model Initialization**

# In[ ]:


# Initialize model
print("\n" + "="*80)
print("MODEL INITIALIZATION")
print("="*80 + "\n")

if USE_DEMO_MODEL:
    print("⚠️ Using demo model (GPT-2) for fast testing")
    print("   For production, set USE_DEMO_MODEL=False to use MedGemma-4B\n")
    model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
else:
    print(f"Loading production model: {model_config.model_name}")
    print("This may take several minutes...\n")
    model = MedicalDigitalTwinModel(model_config, use_demo_model=False)

print(f"✓ Model loaded successfully")
print(f"  Device: {model.device}")
print(f"  Vocabulary size: {len(model.tokenizer)}")


# ## **SECTION 7: Create Training Datasets**

# In[ ]:


# Split into train/eval
train_size = int(0.9 * len(all_training_data))
train_data = all_training_data[:train_size]
eval_data = all_training_data[train_size:]

print(f"\nDataset split:")
print(f"  Training: {len(train_data)} examples")
print(f"  Evaluation: {len(eval_data)} examples")


# In[ ]:


# Create PyTorch datasets
train_dataset = CognitiveStreamDataset(
    train_data,
    model.tokenizer,
    max_length=model_config.max_length
)

eval_dataset = CognitiveStreamDataset(
    eval_data,
    model.tokenizer,
    max_length=model_config.max_length
)

print(f"\n✓ Datasets created")
print(f"  Train dataset: {len(train_dataset)} examples")
print(f"  Eval dataset: {len(eval_dataset)} examples")


# ## **SECTION 8: Phase 1 - Supervised Fine-Tuning (SFT)**

# In[ ]:


# Run SFT training
print("\n" + "="*80)
print("PHASE 1: SUPERVISED FINE-TUNING (SFT)")
print("="*80 + "\n")

if USE_DEMO_MODEL:
    print("Expected time: 5-15 minutes")
else:
    print(f"Expected time: {'30 min - 2 hours' if MAX_PATIENTS <= 100 else '8-24 hours'}")

print("\nStarting SFT training...\n")

sft_result = run_sft_training(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=sft_config
)

print("\n✓ SFT training complete!")
print(f"  Model saved to: {sft_config.output_dir}/final_model")


# In[ ]:


# Clean memory after SFT
clean_memory()
print("✓ Memory cleaned")


# ## **SECTION 9: Phase 4 - GRPO Training (Optional)**

# In[ ]:


# Decide whether to run GRPO
RUN_GRPO = False  # Set to True to run GRPO training

if RUN_GRPO:
    print("\n" + "="*80)
    print("PHASE 4: GROUP RELATIVE POLICY OPTIMIZATION (GRPO)")
    print("="*80 + "\n")

    # Create GRPO dataloader
    from torch.utils.data import DataLoader

    class GRPOPromptDataset:
        def __init__(self, data):
            self.prompts = [item['prompt'] for item in data if 'prompt' in item]

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            return {'prompt': self.prompts[idx]}

    grpo_dataset = GRPOPromptDataset(all_training_data)
    grpo_dataloader = DataLoader(
        grpo_dataset,
        batch_size=grpo_config.batch_size,
        shuffle=True
    )

    print(f"✓ GRPO dataloader created: {len(grpo_dataset)} prompts\n")
    print("Expected time: 12-48 hours\n")

    # Run GRPO training
    run_grpo_training(
        model=model,
        train_dataloader=grpo_dataloader,
        config=grpo_config,
        reward_engine=reward_engine
    )

    print("\n✓ GRPO training complete!")
    print(f"  Model saved to: {grpo_config.output_dir}/final_policy")
else:
    print("\n⏭️  Skipping GRPO training (RUN_GRPO=False)")
    print("   Set RUN_GRPO=True to run reinforcement learning alignment")


# ## **SECTION 10: Evaluation**

# In[ ]:


# Run evaluation
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80 + "\n")

# Initialize evaluator
evaluator = MedicalTwinEvaluator(
    model=model,
    parser=parser,
    reward_engine=reward_engine
)

print("Running evaluation on clinical test cases...\n")

# Run evaluation
results = evaluator.run_evaluation()

# Generate report
report = evaluator.generate_report(results)
print("\n" + report)


# In[ ]:


# Save evaluation results
from datetime import datetime

eval_dir = Path("./evaluation_results")
eval_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_path = eval_dir / f"evaluation_results_{timestamp}.csv"
report_path = eval_dir / f"evaluation_report_{timestamp}.txt"

results.to_csv(results_path, index=False)

with open(report_path, 'w') as f:
    f.write(report)

print(f"\n✓ Results saved to: {results_path}")
print(f"✓ Report saved to: {report_path}")


# ## **SECTION 11: Test Model Interactively**

# In[ ]:


# Test the model with a clinical case
print("\n" + "="*80)
print("INTERACTIVE MODEL TEST")
print("="*80 + "\n")

test_query = """
Patient: 72-year-old male, ICU Day 3
Vital Signs:
- Heart Rate: 125 bpm (was 110)
- Blood Pressure: 85/50 mmHg
- Temperature: 38.9°C

Labs:
- Lactate: 3.2 mmol/L (was 1.8 four hours ago)
- Creatinine: 1.9 mg/dL (baseline 1.0)

Family Question: "Is my father getting better?"
"""

print("Clinical Case:")
print(test_query)
print("\n" + "-"*80)
print("MODEL RESPONSE:")
print("-"*80 + "\n")

# Generate response
response = model.generate(test_query, max_length=512)
print(response)

# Parse and display streams
streams = parser.parse(response)
print("\n" + "-"*80)
print("PARSED COGNITIVE STREAMS:")
print("-"*80)
print(parser.format_for_display(streams))


# ## **SECTION 12: Launch Web Interface (Optional)**

# In[ ]:


# Launch Gradio web interface
LAUNCH_UI = False  # Set to True to launch web interface

if LAUNCH_UI:
    print("\n" + "="*80)
    print("LAUNCHING WEB INTERFACE")
    print("="*80 + "\n")

    from interface.gradio_app import create_gradio_interface

    interface = create_gradio_interface(
        model=model,
        parser=parser,
        reward_engine=reward_engine
    )

    if interface:
        print("✓ Launching Gradio interface...")
        print("  Access at: http://localhost:7860")
        print("  Stop with: Kernel → Interrupt\n")

        interface.launch(
            share=False,  # Set to True for public link
            server_name="0.0.0.0",
            server_port=7860
        )
else:
    print("\n⏭️  Skipping web interface (LAUNCH_UI=False)")
    print("   Set LAUNCH_UI=True to launch interactive interface")


# ## **SECTION 13: Summary and Next Steps**

# In[ ]:


# Print summary
print("\n" + "="*80)
print("PIPELINE COMPLETE - SUMMARY")
print("="*80 + "\n")

print("✓ Completed Steps:")
print(f"  1. Environment setup")
print(f"  2. System tests (all passed)")
print(f"  3. Data preparation ({len(all_training_data)} examples)")
print(f"  4. Model initialization ({model.config.model_name})")
print(f"  5. SFT training (Phase 1)")
if RUN_GRPO:
    print(f"  6. GRPO training (Phase 4)")
print(f"  7. Evaluation")
print(f"  8. Interactive testing")

print("\nOutput Locations:")
print(f"  SFT model: {sft_config.output_dir}/final_model")
if RUN_GRPO:
    print(f"  GRPO model: {grpo_config.output_dir}/final_policy")
print(f"  Evaluation: ./evaluation_results/")
print(f"  Processed data: ./outputs/mimic_processed.json")

print("\nNext Steps:")
print("  • Run evaluation on more test cases")
print("  • Launch web interface for interactive testing")
print("  • Scale up training with more patients")
print("  • Deploy model for clinical validation")

print("\n" + "="*80)
print("Medical Digital Twin Training Complete! 🎉")
print("="*80)

