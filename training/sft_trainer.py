"""
Supervised Fine-Tuning (SFT) Trainer - Phase 1.

Establishes triple-stream cognitive architecture through supervised learning
on MIMIC-IV clinical data + Medical-O1 reasoning chains.

Training Process:
    1. Load MIMIC-IV + Medical-O1 datasets
    2. Format into triple-stream examples
    3. Fine-tune MedGemma-4B with LoRA
    4. Save checkpoints and final model

Expected Time: 8-24 hours on A100 80GB (full training)
                5-15 minutes on A100 80GB (10 patients demo)

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

import logging
import torch
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import inspect

logger = logging.getLogger(__name__)


def _compute_weighted_causal_lm_loss(logits, labels, sample_weights=None):
    """Compute token-masked causal LM loss with optional per-sample weighting."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    per_token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='none'
    ).view(shift_labels.size())

    token_mask = (shift_labels != -100).float()
    per_example_loss = (per_token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)

    if sample_weights is None:
        return per_example_loss.mean()

    weights = sample_weights.to(per_example_loss.device).float().clamp_min(0.0)
    if torch.all(weights == 0):
        return per_example_loss.mean()

    return (per_example_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def run_sft_training(model, train_dataset, eval_dataset, config):
    """
    Run supervised fine-tuning (Phase 1).
    
    Args:
        model: MedicalDigitalTwinModel instance
        train_dataset: CognitiveStreamDataset for training
        eval_dataset: CognitiveStreamDataset for evaluation
        config: SFTConfig with hyperparameters
    
    This phase establishes the triple-stream architecture through
    supervised learning on clinical reasoning examples.
    
    Example:
        >>> from models.mdt_model import MedicalDigitalTwinModel
        >>> from config.configs import ModelConfig, SFTConfig
        >>> 
        >>> model = MedicalDigitalTwinModel(ModelConfig())
        >>> run_sft_training(model, train_dataset, eval_dataset, SFTConfig())
    """
    logger.info("="*80)
    logger.info("STARTING SUPERVISED FINE-TUNING (PHASE 1)")
    logger.info("="*80)
    
    # Validate inputs
    if len(train_dataset) == 0:
        logger.error("❌ Training dataset is empty!")
        return
    
    if len(eval_dataset) == 0:
        logger.warning("⚠️  Evaluation dataset is empty, skipping evaluation")
    
    # Log configuration
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Model: {model.config.model_name}")
    logger.info(f"  Training examples: {len(train_dataset)}")
    logger.info(f"  Evaluation examples: {len(eval_dataset)}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Optimizer: {config.optimizer_type}")
    logger.info(f"  LR scheduler: {config.lr_scheduler_type}")
    logger.info(f"  Mixed precision: {'bf16' if config.bf16 else 'fp16' if config.fp16 else 'fp32'}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info("")
    
    # Create output directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging_dir = Path(config.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training metadata
    metadata = {
        'start_time': datetime.now().isoformat(),
        'model': model.config.model_name,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'config': {
            'epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'optimizer': config.optimizer_type,
        }
    }
    
    with open(output_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Try using Hugging Face Trainer (recommended)
    try:
        from transformers import Trainer, TrainingArguments
        
        logger.info("Using Hugging Face Trainer for SFT...")
        logger.info("")
        
        # Build TrainingArguments with backward-compatible filtering (older transformers may not support some kwargs)
        args_kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": config.num_epochs,
            "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "lr_scheduler_type": config.lr_scheduler_type,
            "logging_dir": str(logging_dir),
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "evaluation_strategy": config.evaluation_strategy if len(eval_dataset) > 0 else "no",
            "eval_steps": config.eval_steps if len(eval_dataset) > 0 else None,
            "fp16": config.fp16,
            "bf16": config.bf16,
            "optim": config.optimizer_type,
            "max_grad_norm": config.max_grad_norm,
            "seed": config.seed,
            "dataloader_num_workers": config.dataloader_num_workers,
            "dataloader_pin_memory": config.dataloader_pin_memory,
            "remove_unused_columns": False,
            "report_to": ["tensorboard"],
            "load_best_model_at_end": True if len(eval_dataset) > 0 else False,
            "metric_for_best_model": "eval_loss" if len(eval_dataset) > 0 else None,
        }

        sig = inspect.signature(TrainingArguments.__init__)
        supported_params = set(sig.parameters.keys())

        # Map evaluation_strategy to eval_strategy if that's the accepted param name
        if "evaluation_strategy" not in supported_params and "eval_strategy" in supported_params:
            args_kwargs["eval_strategy"] = args_kwargs.pop("evaluation_strategy")
        
        filtered_kwargs = {k: v for k, v in args_kwargs.items() if k in supported_params}
        dropped = set(args_kwargs.keys()) - set(filtered_kwargs.keys())
        if dropped:
            logger.warning(f"Dropped unsupported TrainingArguments keys for this transformers version: {sorted(dropped)}")

        # Setup training arguments
        training_args = TrainingArguments(**filtered_kwargs)
        
        class WeightedSFTTrainer(Trainer):
            """Trainer that applies per-sample `think_weight` when available."""

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                think_weight = inputs.pop("think_weight", None)
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                loss = _compute_weighted_causal_lm_loss(outputs.logits, labels, think_weight)
                return (loss, outputs) if return_outputs else loss

        # Initialize trainer
        trainer = WeightedSFTTrainer(
            model=model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        )
        
        # Train!
        logger.info("Starting training...")
        logger.info("")
        
        train_result = trainer.train()
        
        # Save final model
        logger.info("")
        logger.info("Saving final model...")
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        
        # Save training metrics
        metrics = {
            'final_loss': train_result.training_loss,
            'total_steps': train_result.global_step,
            'end_time': datetime.now().isoformat(),
        }
        
        if train_result.metrics:
            metrics.update(train_result.metrics)
        
        with open(output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log final results
        logger.info("")
        logger.info("="*80)
        logger.info("SUPERVISED FINE-TUNING COMPLETE")
        logger.info("="*80)
        logger.info(f"  Final loss: {train_result.training_loss:.4f}")
        logger.info(f"  Total steps: {train_result.global_step}")
        logger.info(f"  Model saved to: {final_model_path}")
        logger.info(f"  Logs saved to: {logging_dir}")
        logger.info("="*80)
        logger.info("")
        
        return model
        
    except ImportError as e:
        logger.warning(f"transformers library not available: {e}")
        logger.warning("Falling back to basic training loop")
        return _basic_training_loop(model, train_dataset, eval_dataset, config, output_dir)
    
    except Exception as e:
        logger.error(f"Trainer failed: {e}", exc_info=True)
        logger.warning("Falling back to basic training loop")
        return _basic_training_loop(model, train_dataset, eval_dataset, config, output_dir)


def _basic_training_loop(model, train_dataset, eval_dataset, config, output_dir):
    """
    Basic training loop implementation (fallback).
    
    Used when Hugging Face Trainer is not available or fails.
    
    Args:
        model: MedicalDigitalTwinModel instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: SFTConfig
        output_dir: Output directory path
    
    Returns:
        Training results dictionary
    """
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    logger.info("Using basic training loop...")
    logger.info("")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        drop_last=True
    )
    
    if len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory
        )
    else:
        eval_loader = None
    
    # Setup optimizer
    if config.optimizer_type == "adamw_torch":
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )
    else:
        # Fallback to standard AdamW
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    # Setup learning rate scheduler
    num_batches = len(train_loader)
    total_steps = (num_batches // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    if config.lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    else:
        scheduler = None
    
    logger.info(f"Training setup:")
    logger.info(f"  Batches per epoch: {num_batches}")
    logger.info(f"  Steps per epoch: {num_batches // config.gradient_accumulation_steps}")
    logger.info(f"  Total training steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info("")
    
    # Training state
    model.model.train()
    global_step = 0
    best_eval_loss = float('inf')
    training_losses = []
    
    # Training loop
    for epoch in range(config.num_epochs):
        logger.info("="*80)
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info("="*80)
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        # Progress bar
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            total=len(train_loader)
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            think_weight = batch.get('think_weight')
            if think_weight is not None:
                think_weight = think_weight.to(model.device)
            
            # Forward pass
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            raw_loss = _compute_weighted_causal_lm_loss(outputs.logits, labels, think_weight)
            loss = raw_loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Track loss
            batch_loss = raw_loss.item()
            epoch_loss += batch_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(),
                    config.max_grad_norm
                )
                
                # Optimizer step
                optimizer.step()
                
                # Learning rate warmup
                if global_step < warmup_steps:
                    lr_scale = min(1.0, float(global_step + 1) / float(warmup_steps))
                    for pg in optimizer.param_groups:
                        pg['lr'] = config.learning_rate * lr_scale
                elif scheduler is not None:
                    scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                # Log periodically
                if global_step % config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    logger.info(f"  Step {global_step}/{total_steps}: "
                              f"loss={avg_loss:.4f}, "
                              f"lr={optimizer.param_groups[0]['lr']:.2e}")
                
                # Save checkpoint periodically
                if global_step % config.save_steps == 0:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    
                    model.model.save_pretrained(str(checkpoint_path))
                    model.tokenizer.save_pretrained(str(checkpoint_path))
                    
                    logger.info(f"  ✓ Saved checkpoint to: {checkpoint_path}")
        
        # Epoch complete
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)
        
        logger.info("")
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Average training loss: {avg_epoch_loss:.4f}")
        
        # Evaluation
        if eval_loader is not None:
            eval_loss = _evaluate(model, eval_loader)
            logger.info(f"  Evaluation loss: {eval_loss:.4f}")
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_path = output_dir / "best_model"
                best_model_path.mkdir(parents=True, exist_ok=True)
                
                model.model.save_pretrained(str(best_model_path))
                model.tokenizer.save_pretrained(str(best_model_path))
                
                logger.info(f"  ✓ New best model saved! (eval_loss: {eval_loss:.4f})")
        
        logger.info("")
    
    # Save final model
    logger.info("Saving final model...")
    final_path = output_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    
    model.model.save_pretrained(str(final_path))
    model.tokenizer.save_pretrained(str(final_path))
    
    # Save training history
    history = {
        'training_losses': training_losses,
        'best_eval_loss': best_eval_loss if eval_loader else None,
        'total_steps': global_step,
        'final_loss': training_losses[-1] if training_losses else None,
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("")
    logger.info("="*80)
    logger.info("SUPERVISED FINE-TUNING COMPLETE")
    logger.info("="*80)
    logger.info(f"  Final training loss: {training_losses[-1]:.4f}")
    if eval_loader:
        logger.info(f"  Best evaluation loss: {best_eval_loss:.4f}")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Model saved to: {final_path}")
    logger.info("="*80)
    logger.info("")
    
    return model


def _evaluate(model, eval_loader):
    """
    Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        eval_loader: DataLoader for evaluation
    
    Returns:
        Average evaluation loss
    """
    model.model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    model.model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def load_sft_checkpoint(model, checkpoint_path):
    """
    Load a saved SFT checkpoint.
    
    Args:
        model: MedicalDigitalTwinModel instance
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        True if loaded successfully, False otherwise
    
    Example:
        >>> model = MedicalDigitalTwinModel(config)
        >>> success = load_sft_checkpoint(model, "./outputs/sft/checkpoint-1000")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load model
        loaded_model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            device_map=model.device,
            torch_dtype=torch.float16 if model.config.load_in_4bit else torch.float32,
        )
        
        # Load tokenizer
        loaded_tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        
        # Update model
        model.model = loaded_model
        model.tokenizer = loaded_tokenizer
        
        logger.info("✓ Checkpoint loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False