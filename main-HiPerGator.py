#!/usr/bin/env python3
"""
Medical Digital Twin - Main Orchestrator (HiPerGator Optimized)

Master file for running all components of the Metacognitive Medical Digital Twin.
Supports both local and HiPerGator environments with MIMIC-IV v3.1.

Author: Ahmed Soliman
Institution: University of Florida, HOBI
"""

import argparse
import logging
import sys
import os
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


def detect_hipergator() -> bool:
    """
    Detect if running on HiPerGator.
    
    Returns:
        True if on HiPerGator, False otherwise
    """
    # Check for SLURM environment variables
    if 'SLURM_JOB_ID' in os.environ:
        return True
    
    # Check for Blue storage path
    cwd = os.getcwd()
    if '/blue/' in cwd or '/orange/' in cwd:
        return True
    
    # Check for HiPerGator hostname
    hostname = os.environ.get('HOSTNAME', '')
    if 'ufhpc' in hostname or 'hpg' in hostname:
        return True
    
    return False


def get_environment_config():
    """
    Get appropriate configuration based on environment.
    
    Returns:
        DataConfig with appropriate paths
    """
    if detect_hipergator():
        try:
            from config.hipergator_config import HiPerGatorConfig
            hpg_config = HiPerGatorConfig()
            logger.info("="*80)
            logger.info("RUNNING ON HIPERGATOR")
            logger.info("="*80)
            logger.info(f"User Group: {hpg_config.user_group}")
            logger.info(f"Username: {hpg_config.username}")
            logger.info(f"MIMIC-IV location: {hpg_config.mimic_root}")
            logger.info(f"Output directory: {hpg_config.output_dir}")
            
            if 'SLURM_JOB_ID' in os.environ:
                logger.info(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
                logger.info(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            
            logger.info("="*80 + "\n")
            
            return hpg_config.get_mimic_config()
        except ImportError:
            logger.warning("HiPerGator detected but config not found. Using default config.")
            logger.warning("Please create config/hipergator_config.py")
            return DataConfig()
    else:
        logger.info("Running on local machine")
        return DataConfig()


def print_banner():
    """Print system banner."""
    print("\n" + "="*80)
    print(" "*20 + "METACOGNITIVE MEDICAL DIGITAL TWIN")
    print(" "*15 + "Complete System Implementation v1.0")
    print("="*80)
    print("\nAuthor: Ahmed Soliman")
    print("Institution: University of Florida, Health Outcomes & Biomedical Informatics")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if detect_hipergator():
        print("\n" + "="*80)
        print("ENVIRONMENT: HiPerGator High-Performance Computing")
        if 'SLURM_JOB_ID' in os.environ:
            print(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
            print(f"Node: {os.environ.get('SLURMD_NODENAME', 'Unknown')}")
            print(f"Partition: {os.environ.get('SLURM_JOB_PARTITION', 'Unknown')}")
        print("="*80)
    
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
        data_config = get_environment_config()
        sft_config = SFTConfig()
        print("✓ All configurations loaded")
        results['configurations'] = True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        results['configurations'] = False
    
    # Test 2: Cognitive Streams
    print("\nTest 2: Cognitive Stream Parser...")
    try:
        parser = CognitiveStreamParser(cog_config)
        test_text = """
        <think>Patient has sepsis with elevated lactate</think>
        <patient_state>HR: 125 bpm [LOINC:8867-4], Lactate: 3.2 mmol/L [LOINC:2524-7]</patient_state>
        <user_belief>Literacy: low, Emotion: anxious</user_belief>
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
    
    # Test 6: Data Processors
    print("\nTest 10: Data Processing...")
    try:
        data_config = get_environment_config()
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
    
    # Test 7: Model
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
    
    # Test 8: End-to-End
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


def handle_train_sft(args):
    """Handle SFT training."""
    print("\n" + "="*80)
    print("SFT TRAINING MODE - PHASE 1")
    print("="*80 + "\n")
    
    # Get environment-specific config
    data_config = get_environment_config()
    mimic_root = Path(data_config.mimic_root_dir)
    
    # Check for data
    if not mimic_root.exists():
        print("❌ MIMIC-IV dataset not found!")
        print(f"   Expected location: {mimic_root}")
        print("\n📥 MIMIC-IV v3.1 Download Instructions:")
        print("   1. Apply for access: https://physionet.org/content/mimiciv/3.1/")
        print("   2. Complete CITI 'Data or Specimens Only Research' training")
        print("   3. Download and extract to the configured path")
        if detect_hipergator():
            print("\n   On HiPerGator:")
            print("   - Update config/hipergator_config.py with your paths")
            print("   - Ensure MIMIC-IV is in your Blue storage")
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
    
    mimic_data = mimic_processor.process_all_patients(
        max_patients=args.max_patients,
        save_path=args.output_dir / "mimic_processed.json" if args.output_dir else None
    )
    print(f"✓ Processed {len(mimic_data)} MIMIC examples")
    
    # Load Medical-O1 data
    try:
        from datasets import load_dataset
        print("\nLoading Medical-O1 reasoning dataset...")
        o1_processor = MedicalO1Processor()
        o1_dataset = o1_processor.load_dataset('train')
        
        if o1_dataset:
            o1_data = o1_processor.format_for_training(
                o1_dataset,
                max_examples=args.max_o1_examples
            )
            print(f"✓ Loaded {len(o1_data)} Medical-O1 examples")
        else:
            print("⚠️  Medical-O1 dataset not available, using MIMIC only")
            o1_data = []
    except ImportError:
        print("⚠️  datasets library not available")
        o1_data = []
    
    # Combine datasets
    all_training_data = mimic_data + o1_data
    print(f"\n✓ Total training examples: {len(all_training_data)}")
    
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
            print("Loading production model (MedGemma-4B)...")
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
        sft_config.output_dir = str(args.output_dir / "sft")
        sft_config.logging_dir = str(args.output_dir / "logs" / "sft")
    
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
    
    # Check for SFT checkpoint
    sft_config = SFTConfig()
    checkpoint_path = Path(sft_config.output_dir)
    
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
    model = MedicalDigitalTwinModel(model_config, use_demo_model=False)
    
    # Load checkpoint
    try:
        model.load_checkpoint(str(checkpoint_path))
        print("✓ SFT model loaded")
    except Exception as e:
        print(f"⚠️  Could not load checkpoint: {e}")
        print("Continuing with base model...")
    
    # Initialize reward engine
    print("\n" + "-"*80)
    print("STEP 2: Initializing Reward Engine")
    print("-"*80 + "\n")
    
    grpo_config = GRPOConfig()
    
    # Update output directory if specified
    if args.output_dir:
        grpo_config.output_dir = str(args.output_dir / "grpo")
    
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
    
    print("⚠️  GRPO training requires prepared dataloader")
    print("Please implement dataloader creation for your specific setup")
    
    # Placeholder - in production, create actual dataloader
    train_dataloader = None
    
    if train_dataloader is None:
        print("\n❌ Training dataloader not configured")
        print("See training/grpo_trainer.py for implementation details")
        return
    
    # Run GRPO training
    print("\n" + "-"*80)
    print("STEP 4: Running GRPO Training")
    print("-"*80 + "\n")
    
    run_grpo_training(model, train_dataloader, grpo_config, reward_engine)
    
    print("\n" + "="*80)
    print("✓ GRPO TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {grpo_config.output_dir}")
    
    # Clean memory
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
    
    # Check if on HiPerGator
    if detect_hipergator() and 'SLURM_JOB_ID' in os.environ:
        print("⚠️  WARNING: Running on HiPerGator compute node")
        print("Web interface may not be accessible externally")
        print("\nRecommendation: Use interactive session on login node")
        print("  srun --partition=hpg-default --mem=16gb --time=04:00:00 --pty bash -i")
        print("\nContinuing anyway...\n")
    
    # Initialize components
    model_config = ModelConfig()
    cog_config = CognitiveArchitectureConfig()
    grpo_config = GRPOConfig()
    
    print("Initializing components...")
    model = MedicalDigitalTwinModel(model_config, use_demo_model=True)
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
        
        if detect_hipergator():
            print("\n📍 Access Instructions:")
            print("   1. Note the URL shown below")
            print("   2. If on compute node, may need SSH tunnel")
            print("   3. Or run on login node for direct access\n")
        else:
            print("   Access at: http://localhost:7860")
        
        print("   Press Ctrl+C to stop\n")
        
        interface.launch(
            share=args.share,
            server_name="0.0.0.0" if detect_hipergator() else "127.0.0.1"
        )
    else:
        print("\n❌ Failed to create interface")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Digital Twin - Complete System (HiPerGator Optimized)",
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
  %(prog)s --launch-ui                  # Web interface (demo model)

HiPerGator:
  sbatch scripts/run_training_hipergator.sh  # Submit SLURM job

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
        "--train-sft",
        action="store_true",
        help="Run supervised fine-tuning (Phase 1)"
    )
    parser.add_argument(
        "--train-grpo",
        action="store_true",
        help="Run GRPO alignment (Phase 4)"
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
        type=Path,
        default=None,
        help="Output directory for results (default: environment-specific)"
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
    
    # Set default output directory based on environment
    if args.output_dir is None:
        if detect_hipergator():
            try:
                from config.hipergator_config import HiPerGatorConfig
                hpg_config = HiPerGatorConfig()
                args.output_dir = Path(hpg_config.output_dir)
            except:
                args.output_dir = Path("./outputs")
        else:
            args.output_dir = Path("./outputs")
    
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
            
            if detect_hipergator():
                print("\nHiPerGator SLURM:")
                print("  sbatch scripts/run_training_hipergator.sh")
            
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