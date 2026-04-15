#!/usr/bin/env python3
"""
Medical Digital Twin - Main Orchestrator

Master file for running all components of the Metacognitive Medical Digital Twin.
Supports testing, training, evaluation, and web interface deployment.

Author: Ahmed Soliman
Institution: University of Florida, Intelligent Clinical Care Center (IC3), Health Outcomes & Biomedical Informatics (HOBI)
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Import configurations
from config.configs import (
    ModelConfig,
    CognitiveArchitectureConfig,
    DataConfig,
    SFTConfig,
    GRPOConfig
)

# Import core modules
from core.cognitive_streams import CognitiveStreamParser
from core.theory_of_mind import TheoryOfMindModule

# Import rewards
from rewards.composite_engine import CompositeRewardEngine

# Import data processing
from data.mimic_processor import MIMICProcessor
from data.medical_o1_processor import MedicalO1Processor
from data.dataset import CognitiveStreamDataset
from data.think_alignment import apply_soft_think_alignment

# Import model
from models.mdt_model import MedicalDigitalTwinModel

# Import training
from training.sft_trainer import run_sft_training
from training.grpo_trainer import run_grpo_training

# Import evaluation
from evaluation.evaluator import MedicalTwinEvaluator

# Import interface
from interface.gradio_app import create_gradio_interface

# Import utilities
from utils.helpers import clean_memory, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def print_banner():
    """Print system banner."""
    print("\n" + "="*80)
    print(" "*20 + "METACOGNITIVE MEDICAL DIGITAL TWIN")
    print(" "*15 + "Complete System Implementation v1.0")
    print("="*80)
    print("\nAuthor: Ahmed Soliman")
    print("Institution: University of Florida, Health Outcomes & Biomedical Informatics")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80 + "\n")


def test_all_components():
    """Run comprehensive system tests."""
    print("\n" + "="*80)
    print(" "*25 + "SYSTEM COMPONENT TESTS")
    print("="*80 + "\n")
    
    results = {}
    
    # Test 1: Configurations
    print("Test 1: Configuration Loading...")
    try:
        model_config = ModelConfig()
        cog_config = CognitiveArchitectureConfig()
        grpo_config = GRPOConfig()
        data_config = DataConfig()
        sft_config = SFTConfig()
        print("✓ All configurations loaded")
        results['configurations'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['configurations'] = False
    
    # Test 2: Cognitive Streams
    print("\nTest 2: Cognitive Stream Parser...")
    try:
        parser = CognitiveStreamParser(cog_config)
        test_text = """
        <think>Patient presents with sepsis based on elevated lactate levels and clinical signs of systemic infection including fever and hypotension requiring immediate intervention</think>
        <patient_state>HR: 125 bpm [LOINC:8867-4], Lactate: 3.2 mmol/L [LOINC:2524-7], Temperature: 38.9°C, BP: 85/50 mmHg</patient_state>
        <user_belief>Literacy: low, Emotional state: anxious, requires simplified medical terminology</user_belief>
        """
        streams = parser.parse(test_text)
        is_valid, error = parser.validate(streams)
        assert is_valid, f"Validation failed: {error}"
        print("✓ Cognitive streams validated")
        results['cognitive_streams'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['cognitive_streams'] = False
    
    # Test 3: Theory of Mind
    print("\nTest 3: Theory of Mind Module...")
    try:
        import torch
        tom = TheoryOfMindModule()
        tom.eval()
        user_emb = torch.randn(1, 768)
        with torch.no_grad():
            outputs = tom(user_emb)
            literacy, conf = tom.infer_literacy_level(user_emb)
        print(f"✓ ToM: {literacy.value} (confidence: {conf:.2f})")
        results['theory_of_mind'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['theory_of_mind'] = False
    
    # Test 4: Reward Components
    print("\nTest 4-8: Reward Components...")
    try:
        import torch
        from rewards.semantic_reward import SemanticFidelityReward
        from rewards.metacognitive_reward import MetacognitiveDepthReward
        from rewards.empathy_reward import StructuralEmpathyReward
        from rewards.proactivity_reward import ProactivityReward
        from rewards.safety_reward import BiologicalSafetyReward
        
        # Test each reward
        sem = SemanticFidelityReward()
        meta = MetacognitiveDepthReward()
        emp = StructuralEmpathyReward()
        pro = ProactivityReward()
        safety = BiologicalSafetyReward()
        
        # Quick test
        sem_score = sem.compute(["Patient has sepsis"], ["Patient presents with sepsis"])
        meta_score = meta.compute(["Dehydration"], ["Actually sepsis"])
        
        print(f"✓ All reward components initialized")
        print(f"  - Semantic: {sem_score[0]:.4f}")
        print(f"  - Metacognitive: {meta_score[0]:.4f}")
        results['rewards'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['rewards'] = False
    
    # Test 5: Composite Engine
    print("\nTest 9: Composite Reward Engine...")
    try:
        engine = CompositeRewardEngine()
        print("✓ Composite engine initialized")
        results['composite_engine'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['composite_engine'] = False
    
    # Test 6: Ontology Validation
    print("\nTest 9.5: Ontology Validation...")
    try:
        from utils.ontology_validator import OntologyValidator
        from config.ontology import LOINCCodes, SNOMEDCodes, ClinicalReferenceRanges
        
        validator = OntologyValidator()
        
        # Test LOINC codes
        hr_code = LOINCCodes.get_code_by_name('heart_rate')
        assert hr_code == '8867-4', f"Expected '8867-4', got '{hr_code}'"
        
        # Test SNOMED codes
        sepsis_code = SNOMEDCodes.get_code_by_name('sepsis')
        assert sepsis_code == '91302008', f"Expected '91302008', got '{sepsis_code}'"
        
        # Test reference ranges
        hr_range = ClinicalReferenceRanges.get_range('heart_rate')
        assert hr_range is not None, "Heart rate reference range not found"
        assert hr_range.is_normal(75), "75 bpm should be normal"
        assert not hr_range.is_normal(150), "150 bpm should be abnormal"
        assert hr_range.is_critical(150), "150 bpm should be critical"
        
        # Test patient state validation
        test_state = "HR: 125 bpm [LOINC:8867-4], BP: 85/50 mmHg [LOINC:8480-6]"
        validations = validator.extract_and_validate_patient_state(test_state)
        
        print("✓ Ontology validation working")
        print(f"  - LOINC codes: {len(LOINCCodes.get_all_codes())}")
        print(f"  - SNOMED codes: {len(SNOMEDCodes.get_all_codes())}")
        print(f"  - Reference ranges: {len(ClinicalReferenceRanges.RANGES)}")
        results['ontology'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['ontology'] = False
    
    # Test 7: Data Processors
    print("\nTest 10: Data Processing...")
    try:
        data_config = DataConfig()
        mimic_processor = MIMICProcessor(data_config)
        o1_processor = MedicalO1Processor()
        
        # Check MIMIC availability
        mimic_available = mimic_processor.check_availability()
        print(f"✓ Data processors initialized")
        print(f"  - MIMIC-IV v{data_config.mimic_version} available: {mimic_available}")
        
        results['data_processing'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['data_processing'] = False
    
    # Test 8: Model
    print("\nTest 11: Model Initialization...")
    try:
        import torch
        model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
        print(f"✓ Model initialized on {model.device}")
        print(f"  - Model type: Demo (GPT-2)")
        print(f"  - Vocab size: {len(model.tokenizer)}")
        results['model'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['model'] = False
    
    # Test 9: End-to-End
    print("\nTest 12: End-to-End Inference...")
    try:
        response = model.generate("Patient with fever", max_length=256)
        print(f"✓ Generated {len(response)} chars")
        results['end_to_end'] = True
    except Exception as e:
        print(f"✗ Failed: {e}")
        results['end_to_end'] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*80 + "\n")
    
    return results


def run_demo():
    """Run system demonstration."""
    print("\n" + "="*80)
    print("CLINICAL CASE DEMONSTRATION")
    print("="*80 + "\n")
    
    clinical_case = """
Patient: 72-year-old male, ICU Day 3
Admission: Septic shock secondary to pneumonia

Current Status:
- Heart Rate: 125 bpm (↑ from 110)
- Blood Pressure: 85/50 mmHg (on norepinephrine)
- SpO2: 94% on 4L O2
- Temperature: 38.9°C

Labs:
- Lactate: 3.2 mmol/L (was 1.8 four hours ago) ⚠
- Creatinine: 1.9 mg/dL (baseline 1.0) ⚠
- WBC: 18,000

Family Question: "Is my father getting better?"
"""
    
    print(clinical_case)
    
    print("\n" + "-"*80)
    print("EXPECTED MDT RESPONSE STRUCTURE:")
    print("-"*80 + "\n")
    
    expected = """
<think>
Clinical Assessment:
- Hemodynamics deteriorating: HR↑, persistent hypotension
- Metabolic: Lactate doubled (1.8→3.2) = worsening hypoperfusion
- Renal: Creatinine rising = AKI Stage 2
- This represents DETERIORATION requiring escalation
</think>

<patient_state>
- Heart Rate: 125 bpm [LOINC:8867-4] - Tachycardia
- Lactate: 3.2 mmol/L [LOINC:2524-7] - CRITICAL
  * Doubled in 4h
  * Trajectory: May reach 4.0+ if trend continues
- Creatinine: 1.9 mg/dL [LOINC:2160-0] - AKI Stage 2
</patient_state>

<user_belief>
- Relationship: Family member (daughter)
- Literacy: LOW-MEDIUM
- Emotional State: HIGHLY ANXIOUS
- Strategy: Use simple language, be honest but compassionate
</user_belief>

Response:
I appreciate you asking. I want to be honest - he is not improving right now.
We're seeing concerning changes that tell us the infection is putting more
stress on his body...
"""
    
    print(expected)


def create_grpo_dataloader(prompts_file: str, batch_size: int = 32):
    """
    Create GRPO training dataloader from processed prompts.
    
    Args:
        prompts_file: Path to JSON file with processed prompts
        batch_size: Batch size for GRPO training
    
    Returns:
        DataLoader or None if failed
    """
    try:
        from torch.utils.data import Dataset, DataLoader
        
        class GRPOPromptDataset(Dataset):
            """Dataset of clinical prompts for GRPO training."""
            
            def __init__(self, prompts_file: str):
                """Load prompts from file."""
                with open(prompts_file, 'r') as f:
                    data = json.load(f)
                
                # Extract prompts
                if isinstance(data, list):
                    self.prompts = [item['prompt'] for item in data if 'prompt' in item]
                else:
                    self.prompts = []
                
                logger.info(f"Loaded {len(self.prompts)} prompts for GRPO training")
            
            def __len__(self):
                return len(self.prompts)
            
            def __getitem__(self, idx):
                return {'prompt': self.prompts[idx]}
        
        # Create dataset
        dataset = GRPOPromptDataset(prompts_file)
        
        if len(dataset) == 0:
            logger.error("No prompts found in file")
            return None
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Created GRPO dataloader with {len(dataset)} prompts, batch size {batch_size}")
        return dataloader
        
    except Exception as e:
        logger.error(f"Failed to create GRPO dataloader: {e}")
        return None


def handle_train_sft(args):
    """Handle SFT training."""
    print("\n" + "="*80)
    print("SFT TRAINING MODE - PHASE 1")
    print("="*80 + "\n")
    
    # Get configuration
    data_config = DataConfig()
    mimic_root = Path(data_config.mimic_root_dir)
    
    # Check for data
    if not mimic_root.exists():
        print("❌ MIMIC-IV dataset not found!")
        print(f"   Expected location: {mimic_root}")
        print("\n📥 MIMIC-IV v3.1 Download Instructions:")
        print("   1. Apply for access: https://physionet.org/content/mimiciv/3.1/")
        print("   2. Complete CITI 'Data or Specimens Only Research' training")
        print("   3. Download and extract to the configured path")
        print(f"\n   Configure the path in: config/configs.py")
        print(f"   Current setting: mimic_root_dir = '{data_config.mimic_root_dir}'")
        print("\n⚠️  Cannot proceed without data.")
        return
    
    print("✓ MIMIC-IV dataset found")
    print(f"  Location: {mimic_root}")
    print(f"  Version: {data_config.mimic_version}")
    
    print("\n" + "-"*80)
    print("STEP 1: Processing Training Data")
    print("-"*80 + "\n")
    
    # Process MIMIC-IV data
    print("Processing MIMIC-IV ICU data...")
    mimic_processor = MIMICProcessor(data_config)
    
    # Check availability first
    if not mimic_processor.check_availability():
        print("❌ MIMIC-IV files not accessible")
        return
    
    save_path = Path(args.output_dir) / "mimic_processed.json" if args.output_dir else None
    
    mimic_data = mimic_processor.process_all_patients(
        max_patients=args.max_patients,
        save_path=str(save_path) if save_path else None
    )
    mimic_data = [{**item, 'source': 'mimic'} for item in mimic_data]
    print(f"✓ Processed {len(mimic_data)} MIMIC examples")
    
    # Load Medical-O1 data
    try:
        print("\nLoading Medical-O1 reasoning dataset...")
        o1_processor = MedicalO1Processor()
        
        # Use config from DataConfig
        o1_dataset = o1_processor.load_dataset('train', config=data_config.medical_o1_config)
        
        if o1_dataset:
            o1_data = o1_processor.format_for_training(
                o1_dataset,
                max_examples=args.max_o1_examples
            )
            o1_data = [{**item, 'source': 'medical_o1'} for item in o1_data]
            print(f"✓ Loaded {len(o1_data)} Medical-O1 examples")
        else:
            print("⚠️  Medical-O1 dataset not available, using MIMIC only")
            o1_data = []
    except ImportError:
        print("⚠️  datasets library not available")
        print("   Install with: pip install datasets")
        o1_data = []
    except Exception as e:
        print(f"⚠️  Medical-O1 loading failed: {e}")
        print("   Continuing with MIMIC data only")
        o1_data = []


    # Combine all datasets
    all_training_data = mimic_data + o1_data

    if data_config.soft_think_enabled:
        all_training_data, alignment_summary = apply_soft_think_alignment(
            all_training_data,
            config=data_config
        )
        print("✓ Applied soft-mandatory CoT alignment")
        print(f"  Gold think samples: {alignment_summary['gold_examples']}")
        print(f"  Synthetic think samples: {alignment_summary['synthetic_examples']}")
        print(f"  Synthetic quality pass rate: {alignment_summary['quality_pass_rate']:.2%}")
        print(f"  Average think weight: {alignment_summary['avg_think_weight']:.3f}")
        print(f"  Teacher model tag: {alignment_summary['teacher_model']}")
    
    print(f"\n✓ Total training examples: {len(all_training_data)}")
    print(f"  MIMIC-IV: {len(mimic_data)}")
    print(f"  Medical-O1: {len(o1_data)}")
    

    
    if len(all_training_data) == 0:
        print("\n❌ No training data available. Exiting.")
        return
    
    print("\n" + "-"*80)
    print("STEP 2: Initializing Model")
    print("-"*80 + "\n")
    
    # Initialize model
    model_config = ModelConfig()
    print(f"Loading model: {model_config.model_name}")
    
    if args.use_demo:
        print("⚠️  Using demo model (GPT-2) for testing")
        model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
    else:
        try:
            print("Loading production model (MedGemma-1.5-4B-IT)...")
            model = MedicalDigitalTwinModel(model_config, use_demo_model=False)
            print("✓ Production model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load production model: {e}")
            print("\n💡 Falling back to demo model...")
            model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
    
    # Create datasets
    print("\n" + "-"*80)
    print("STEP 3: Creating Training Datasets")
    print("-"*80 + "\n")
    
    train_size = int(0.9 * len(all_training_data))
    train_data = all_training_data[:train_size]
    eval_data = all_training_data[train_size:]
    
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
    
    print(f"✓ Training set: {len(train_dataset)} examples")
    print(f"✓ Evaluation set: {len(eval_dataset)} examples")
    
    # Training configuration
    sft_config = SFTConfig()
    
    # Update output directory if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        sft_config.output_dir = str(output_path / "sft")
        sft_config.logging_dir = str(output_path / "logs" / "sft")
    
    print("\n" + "-"*80)
    print("STEP 4: Running SFT Training")
    print("-"*80 + "\n")
    
    run_sft_training(model, train_dataset, eval_dataset, sft_config)
    
    print("\n" + "="*80)
    print("✓ SFT TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {sft_config.output_dir}")
    
    # Clean memory
    clean_memory()


def handle_train_grpo(args):
    """Handle GRPO training."""
    print("\n" + "="*80)
    print("GRPO TRAINING MODE - PHASE 4")
    print("="*80 + "\n")
    
    # Check for SFT checkpoint (respect --output-dir when provided)
    sft_config = SFTConfig()
    checkpoint_path = Path(args.output_dir) / "sft" if args.output_dir else Path(sft_config.output_dir)
    
    if not checkpoint_path.exists():
        print("❌ No SFT checkpoint found!")
        print(f"   Expected location: {checkpoint_path}")
        print("\n💡 You must run SFT training first:")
        print("   python main.py --train-sft")
        return
    
    print(f"✓ Found SFT checkpoint at: {checkpoint_path}")
    
    # Load model
    print("\n" + "-"*80)
    print("STEP 1: Loading SFT Model")
    print("-"*80 + "\n")
    
    model_config = ModelConfig()
    model = MedicalDigitalTwinModel(model_config, use_demo_model=bool(args.use_demo))

    # Prepare GRPO config early (needed for resume path resolution)
    grpo_config = GRPOConfig()
    if args.output_dir:
        output_path = Path(args.output_dir)
        grpo_config.output_dir = str(output_path / "grpo")

    if args.use_demo:
        # Keep demo smoke runs fast and within GPT-2 context/compute limits.
        grpo_config.sanity_check_mode = True
        grpo_config.generation_max_length = min(int(grpo_config.generation_max_length), 256)

    if args.grpo_iterations is not None:
        grpo_config.num_iterations = args.grpo_iterations
    
    # Load checkpoint (SFT default; optionally resume from latest GRPO checkpoint)
    load_path = checkpoint_path
    if args.resume_grpo:
        grpo_output = Path(grpo_config.output_dir)
        candidates = []
        if grpo_output.exists():
            for child in grpo_output.iterdir():
                if child.is_dir() and child.name.startswith("iteration_"):
                    try:
                        iteration_num = int(child.name.replace("iteration_", "", 1))
                        candidates.append((iteration_num, child))
                    except ValueError:
                        continue
        if candidates:
            candidates.sort(key=lambda x: x[0])
            load_path = candidates[-1][1]
            print(f"✓ Resuming from latest GRPO checkpoint: {load_path}")
        else:
            print("⚠️  --resume-grpo set but no GRPO checkpoints found; using SFT checkpoint")

    try:
        model.load_checkpoint(str(load_path))
        print(f"✓ Model checkpoint loaded from: {load_path}")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        print("Continuing with base model...")
    
    # Initialize reward engine
    print("\n" + "-"*80)
    print("STEP 2: Initializing Reward Engine")
    print("-"*80 + "\n")
    
    # GRPO config already initialized above (with optional overrides)
    
    reward_engine = CompositeRewardEngine(
        w_semantic=grpo_config.w_semantic,
        w_metacognitive=grpo_config.w_metacognitive,
        w_empathy=grpo_config.w_empathy,
        w_proactivity=grpo_config.w_proactivity,
        w_safety=grpo_config.w_safety
    )
    print("✓ Reward engine initialized")
    print(f"   Weights: sem={grpo_config.w_semantic}, meta={grpo_config.w_metacognitive}, "
          f"emp={grpo_config.w_empathy}, physio={grpo_config.w_proactivity}, "
          f"safety={grpo_config.w_safety}")
    
    # Load training data
    print("\n" + "-"*80)
    print("STEP 3: Loading Training Data")
    print("-"*80 + "\n")
    
    # Try to create GRPO dataloader from processed MIMIC data
    prompts_file = Path(args.output_dir) / "mimic_processed.json" if args.output_dir else Path("./outputs/mimic_processed.json")
    
    if prompts_file.exists():
        print(f"Found processed prompts at: {prompts_file}")
        train_dataloader = create_grpo_dataloader(
            str(prompts_file),
            batch_size=grpo_config.batch_size
        )
        
        if train_dataloader:
            print(f"✓ Created GRPO dataloader with {len(train_dataloader.dataset)} prompts")
        else:
            print("❌ Failed to create dataloader from prompts file")
            train_dataloader = None
    else:
        print(f"⚠️  Prompts file not found: {prompts_file}")
        print("   Run SFT training first to generate prompts:")
        print("   python main.py --train-sft --max-patients 100")
        train_dataloader = None
    
    if train_dataloader is None:
        print("\n" + "="*80)
        print("GRPO DATALOADER SETUP REQUIRED")
        print("="*80)
        print("\nTo proceed with GRPO training, you need:")
        print("  1. Processed prompts file (created during SFT training)")
        print("  2. Or manually create prompts file with format:")
        print("     [{'prompt': 'Clinical case...', ...}, ...]")
        print("\nYour SFT model is already trained and ready to use!")
        print("\nNext steps:")
        print("  • Evaluate model: python main.py --evaluate")
        print("  • Launch UI: python main.py --launch-ui")
        print("  • Re-run SFT to generate prompts: python main.py --train-sft")
        print("")
        return
    
    # Run GRPO training
    print("\n" + "-"*80)
    print("STEP 4: Running GRPO Training")
    print("-"*80 + "\n")

    print(f"GRPO output dir: {grpo_config.output_dir}")
    print(f"Target iterations: {grpo_config.num_iterations}")
    print(f"Resume mode: {args.resume_grpo}")
    
    run_grpo_training(model, train_dataloader, grpo_config, reward_engine)
    
    print("\n" + "="*80)
    print("✓ GRPO TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {grpo_config.output_dir}")
    
    # Clean memory
    clean_memory()


def handle_sanity_sft(args):
    """Run a tiny in-memory SFT sanity check for soft-CoT weighting path."""
    print("\n" + "="*80)
    print("SFT SANITY MODE - SOFT COT WEIGHTING")
    print("="*80 + "\n")

    output_root = Path(args.output_dir) if args.output_dir else Path("./outputs/sanity_softcot_cli")
    sft_output = output_root / "sft"
    sft_logs = output_root / "logs" / "sft"

    examples = [
        {
            'source': 'mimic',
            'case_description': 'ICU patient with fever and hypotension',
            'prompt': 'Assess ICU patient with fever and hypotension',
            'patient_state': 'HR 122, SBP 86, Temp 101.2F',
            'user_belief': 'Family member anxious',
            'response': 'We are treating probable sepsis and monitoring closely.'
        },
        {
            'source': 'mimic',
            'case_description': 'ICU patient with dyspnea and low oxygen',
            'prompt': 'Assess respiratory decline',
            'patient_state': 'SpO2 88%, RR 30',
            'user_belief': 'Nurse handoff',
            'response': 'Escalate oxygen support and evaluate cause immediately.'
        },
        {
            'source': 'medical_o1',
            'prompt': 'How to evaluate sepsis progression?',
            'think': 'Evaluate infection source, lactate trend, hemodynamics, and organ dysfunction markers.',
            'patient_state': 'Lactate rising to 3.5',
            'user_belief': 'Medical trainee',
            'response': 'Prioritize source control, fluids, and serial reassessment.'
        },
        {
            'source': 'medical_o1',
            'prompt': 'Best first steps in shock management?',
            'think': 'Assess airway breathing circulation, perfusion markers, and likely etiology before targeted therapy.',
            'patient_state': 'MAP 58, tachycardia',
            'user_belief': 'Resident physician',
            'response': 'Start stabilization bundle and etiology-directed treatment quickly.'
        }
    ]

    data_cfg = DataConfig()
    aligned, summary = apply_soft_think_alignment(examples, data_cfg)
    print(f"ALIGN_SUMMARY {summary}")

    model_cfg = ModelConfig()
    model = MedicalDigitalTwinModel(model_cfg, use_demo_model=True)

    train_dataset = CognitiveStreamDataset(aligned[:3], model.tokenizer, max_length=min(256, model_cfg.max_length))
    eval_dataset = CognitiveStreamDataset(aligned[3:], model.tokenizer, max_length=min(256, model_cfg.max_length))

    sft_cfg = SFTConfig()
    sft_cfg.num_epochs = 1
    sft_cfg.batch_size = 1
    sft_cfg.gradient_accumulation_steps = 1
    sft_cfg.logging_steps = 1
    sft_cfg.save_steps = 1000
    sft_cfg.eval_steps = 1000
    sft_cfg.dataloader_num_workers = 0
    sft_cfg.fp16 = False
    sft_cfg.bf16 = False
    sft_cfg.output_dir = str(sft_output)
    sft_cfg.logging_dir = str(sft_logs)

    run_sft_training(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, config=sft_cfg)

    print("\n✓ SFT sanity run complete")
    print(f"  Output dir: {sft_output}")
    print("  Marker: SFT_SANITY_DONE")

    clean_memory()


def handle_evaluate(args):
    """Handle evaluation."""
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80 + "\n")
    
    # Initialize components
    model_config = ModelConfig()
    cog_config = CognitiveArchitectureConfig()
    grpo_config = GRPOConfig()
    
    print("Initializing components...")
    
    # Load model
    if args.use_demo:
        print("Using demo model (GPT-2)")
        model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
    else:
        print("Loading production model...")
        model = MedicalDigitalTwinModel(model_config, use_demo_model=False)
        
        # Try to load trained checkpoint if available
        checkpoint_path = Path("./outputs/grpo")
        if checkpoint_path.exists():
            try:
                model.load_checkpoint(str(checkpoint_path))
                print("✓ Loaded trained model")
            except:
                print("⚠️  Using base model")
    
    parser = CognitiveStreamParser(cog_config)
    reward_engine = CompositeRewardEngine(
        w_semantic=grpo_config.w_semantic,
        w_metacognitive=grpo_config.w_metacognitive,
        w_empathy=grpo_config.w_empathy,
        w_proactivity=grpo_config.w_proactivity,
        w_safety=grpo_config.w_safety
    )
    
    # Run evaluation
    evaluator = MedicalTwinEvaluator(model, parser, reward_engine)
    results = evaluator.run_evaluation()
    
    # Generate report
    report = evaluator.generate_report(results)
    print("\n" + report)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path("./evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    results.to_csv(results_path, index=False)
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Report saved to: {report_path}")
    
    # Clean memory
    clean_memory()


def handle_launch_ui(args):
    """Handle UI launch."""
    print("\n" + "="*80)
    print("LAUNCHING WEB INTERFACE")
    print("="*80 + "\n")
    
    # Initialize components
    model_config = ModelConfig()
    cog_config = CognitiveArchitectureConfig()
    grpo_config = GRPOConfig()
    
    use_demo = not args.use_prod
    
    print(f"Initializing components (Demo Model: {use_demo})...")
    model = MedicalDigitalTwinModel(model_config, use_demo_model=use_demo)
    
    if not use_demo and hasattr(args, 'checkpoint_path'):
        try:
            model.load_checkpoint(args.checkpoint_path)
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {args.checkpoint_path}: {e}")
            
    parser = CognitiveStreamParser(cog_config)
    engine = CompositeRewardEngine(
        w_semantic=grpo_config.w_semantic,
        w_metacognitive=grpo_config.w_metacognitive,
        w_empathy=grpo_config.w_empathy,
        w_proactivity=grpo_config.w_proactivity,
        w_safety=grpo_config.w_safety
    )
    
    # Create and launch interface
    interface = create_gradio_interface(model, parser, engine)
    
    if interface:
        print("\n✓ Launching web interface...")
        print("   Access at: http://localhost:7860")
        print("   Press Ctrl+C to stop\n")
        
        interface.launch(share=args.share)
    else:
        print("\n❌ Failed to create interface")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Digital Twin - Complete System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended Workflow:
  1. %(prog)s --test-only              # Verify system (5 min)
  2. Download MIMIC-IV v3.1             # One-time setup
  3. %(prog)s --train-sft               # Phase 1: SFT training (hours-days)
  4. %(prog)s --train-grpo              # Phase 4: GRPO alignment (hours-days)
  5. %(prog)s --evaluate                # Evaluate trained model
  6. %(prog)s --launch-ui               # Launch web interface

Quick Commands:
  %(prog)s                              # Run demo and tests (no training)
  %(prog)s --test-only                  # Tests only
    %(prog)s --sanity-sft                 # Tiny soft-CoT weighted SFT smoke test
  %(prog)s --launch-ui                  # Web interface (demo model)

For more information, visit: https://github.com/ahmedsoliman/medical-digital-twin
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run comprehensive tests only"
    )
    parser.add_argument(
        "--launch-ui",
        action="store_true",
        help="Launch Gradio web interface"
    )
    parser.add_argument(
        "--use-prod",
        action="store_true",
        help="Use production model instead of demo model for UI or Eval"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="outputs/sft",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--train-sft",
        action="store_true",
        help="Run supervised fine-tuning (Phase 1)"
    )
    parser.add_argument(
        "--sanity-sft",
        action="store_true",
        help="Run tiny in-memory SFT sanity check for soft-CoT weighting"
    )
    parser.add_argument(
        "--train-grpo",
        action="store_true",
        help="Run GRPO alignment (Phase 4)"
    )
    parser.add_argument(
        "--resume-grpo",
        action="store_true",
        help="Resume GRPO from latest iteration checkpoint in output dir"
    )
    parser.add_argument(
        "--grpo-iterations",
        type=int,
        default=None,
        help="Override GRPO total target iterations (useful for walltime chunking)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation suite"
    )
    
    # Training parameters
    parser.add_argument(
        "--max-patients",
        type=int,
        default=1000,
        help="Maximum MIMIC patients to process (default: 1000)"
    )
    parser.add_argument(
        "--max-o1-examples",
        type=int,
        default=5000,
        help="Maximum Medical-O1 examples to load (default: 5000)"
    )
    parser.add_argument(
        "--use-demo",
        action="store_true",
        help="Use demo model (GPT-2) instead of production model"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for results (default: ./outputs)"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Compare trained model against base model"
    )
    
    # Interface parameters
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create shareable Gradio link"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle different modes
    try:
        if args.test_only:
            test_all_components()
        
        elif args.launch_ui:
            handle_launch_ui(args)
        
        elif args.train_sft:
            handle_train_sft(args)

        elif args.sanity_sft:
            handle_sanity_sft(args)
        
        elif args.train_grpo:
            handle_train_grpo(args)
        
        elif args.evaluate:
            handle_evaluate(args)
        
        else:
            # Default: Run demo and tests
            test_all_components()
            run_demo()
            
            print("\n" + "="*80)
            print("QUICK START COMMANDS")
            print("="*80 + "\n")
            print("Run tests:")
            print("  python main.py --test-only")
            print("\nLaunch web interface:")
            print("  python main.py --launch-ui")
            print("\nRun evaluation:")
            print("  python main.py --evaluate")
            print("\nTrain model:")
            print("  python main.py --train-sft --max-patients 100")
            print("  python main.py --train-grpo")
            print("\n" + "="*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting gracefully...")
        clean_memory()
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        clean_memory()
        sys.exit(1)


if __name__ == "__main__":
    main()