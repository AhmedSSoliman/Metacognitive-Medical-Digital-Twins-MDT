"""
Group Relative Policy Optimization (GRPO) Trainer - Phase 4.

Implements multi-objective reinforcement learning alignment using:
    - Group Relative Policy Optimization (GRPO)
    - Composite reward function (5 components)
    - KL-divergence constraint
    - Advantage estimation

Mathematical Formulation:
    L_GRPO = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)
    
    where:
    - r(θ) = π_θ(a|s) / π_ref(a|s) (probability ratio)
    - A = advantage (computed from group-relative rewards)
    - ε = clip_range (0.2)
    - β = KL penalty coefficient

Expected Time: 12-48 hours on A100 80GB GPU

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

import logging
from copy import deepcopy
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def _save_grpo_artifacts(output_dir: Path, training_stats: Dict[str, List[float]]) -> None:
    """Persist GRPO stats to JSON/CSV and plot metrics figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON snapshot
    with open(output_dir / 'training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)

    # CSV snapshot
    csv_path = output_dir / 'training_stats.csv'
    keys = list(training_stats.keys())
    row_count = len(training_stats.get('iterations', []))
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for idx in range(row_count):
            row = []
            for key in keys:
                values = training_stats.get(key, [])
                row.append(str(values[idx]) if idx < len(values) else '')
            f.write(','.join(row) + '\n')

    # Plot snapshot (best-effort)
    try:
        import matplotlib.pyplot as plt

        x = training_stats.get('iterations', [])
        if not x:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0, 0].plot(x, training_stats.get('rewards', []), label='Total Reward', color='tab:blue')
        axes[0, 0].set_title('Average Total Reward')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(alpha=0.25)

        axes[0, 1].plot(x, training_stats.get('policy_losses', []), label='Policy Loss', color='tab:red')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(alpha=0.25)

        axes[1, 0].plot(x, training_stats.get('kl_divergences', []), label='KL Divergence', color='tab:green')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('KL')
        axes[1, 0].grid(alpha=0.25)

        axes[1, 1].plot(x, training_stats.get('semantic_rewards', []), label='Semantic', alpha=0.9)
        axes[1, 1].plot(x, training_stats.get('metacognitive_rewards', []), label='Metacognitive', alpha=0.9)
        axes[1, 1].plot(x, training_stats.get('empathy_rewards', []), label='Empathy', alpha=0.9)
        axes[1, 1].plot(x, training_stats.get('proactivity_rewards', []), label='Proactivity', alpha=0.9)
        axes[1, 1].plot(x, training_stats.get('safety_rewards', []), label='Safety', alpha=0.9)
        axes[1, 1].set_title('Reward Components')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Component Score')
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / 'training_curves.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not create GRPO training plot: {e}")


def _find_latest_grpo_iteration(output_dir: Path) -> int:
    """Find latest completed GRPO iteration from checkpoint dirs and stats."""
    latest = 0

    # From checkpoint folders
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and child.name.startswith("iteration_"):
                raw = child.name.replace("iteration_", "", 1)
                try:
                    latest = max(latest, int(raw))
                except ValueError:
                    continue

    # From stats length
    stats_path = output_dir / 'training_stats.json'
    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            iter_list = stats.get('iterations', [])
            if iter_list:
                latest = max(latest, int(iter_list[-1]))
        except Exception:
            pass

    return latest


def run_grpo_training(
    model,
    train_dataloader,
    config,
    reward_engine
):
    """
    Run GRPO training (Phase 4).
    
    This implementation provides a complete GRPO training loop with:
        1. Policy rollouts (generate K responses per prompt)
        2. Reward computation for each response
        3. Group-relative advantage estimation
        4. PPO-style policy updates with clipping
        5. KL divergence constraint
    
    Args:
        model: MedicalDigitalTwinModel (from SFT)
        train_dataloader: DataLoader with prompts
        config: GRPOConfig with hyperparameters
        reward_engine: CompositeRewardEngine for multi-objective rewards
    
    Example:
        >>> from models.mdt_model import MedicalDigitalTwinModel
        >>> from config.configs import GRPOConfig
        >>> from rewards.composite_engine import CompositeRewardEngine
        >>> 
        >>> model = MedicalDigitalTwinModel(config)
        >>> reward_engine = CompositeRewardEngine()
        >>> run_grpo_training(model, dataloader, GRPOConfig(), reward_engine)
    """
    logger.info("="*80)
    logger.info("STARTING GROUP RELATIVE POLICY OPTIMIZATION (PHASE 4)")
    logger.info("="*80)
    
    # Configuration logging
    logger.info("")
    logger.info("GRPO Configuration:")
    logger.info(f"  Iterations: {config.num_iterations}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  K generations per prompt: {config.num_generations_per_prompt}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Clip range: {config.clip_range}")
    logger.info(f"  Target KL: {config.target_kl}")
    logger.info(f"  GAE lambda: {config.gae_lambda}")
    logger.info("")

    # Optional fast sanity mode (backward-compatible via getattr)
    # Add these attributes to GRPOConfig at runtime if desired:
    #   sanity_check_mode: bool
    #   sanity_num_iterations: int
    #   sanity_num_generations_per_prompt: int
    #   sanity_max_prompts_per_batch: int
    sanity_mode = bool(getattr(config, "sanity_check_mode", False))
    requested_num_iterations = config.num_iterations
    requested_num_generations = config.num_generations_per_prompt
    max_prompts_per_batch = max(1, int(getattr(config, "sanity_max_prompts_per_batch", config.batch_size)))

    if sanity_mode:
        sanity_iters = max(1, int(getattr(config, "sanity_num_iterations", 2)))
        sanity_gens = max(1, int(getattr(config, "sanity_num_generations_per_prompt", 2)))
        config.num_iterations = min(config.num_iterations, sanity_iters)
        config.num_generations_per_prompt = min(config.num_generations_per_prompt, sanity_gens)

        logger.info("SANITY MODE ENABLED")
        logger.info(f"  Iterations: {requested_num_iterations} -> {config.num_iterations}")
        logger.info(f"  Generations/prompt: {requested_num_generations} -> {config.num_generations_per_prompt}")
        logger.info(f"  Max prompts per batch: {max_prompts_per_batch}")
        logger.info("")
    logger.info("Reward Weights:")
    logger.info(f"  Semantic: {config.w_semantic}")
    logger.info(f"  Metacognitive: {config.w_metacognitive}")
    logger.info(f"  Empathy: {config.w_empathy}")
    logger.info(f"  Proactivity: {config.w_proactivity}")
    logger.info(f"  Safety: {config.w_safety}")
    logger.info("")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = {
        'num_iterations': config.num_iterations,
        'batch_size': config.batch_size,
        'num_generations_per_prompt': config.num_generations_per_prompt,
        'learning_rate': config.learning_rate,
        'clip_range': config.clip_range,
        'reward_weights': {
            'semantic': config.w_semantic,
            'metacognitive': config.w_metacognitive,
            'empathy': config.w_empathy,
            'proactivity': config.w_proactivity,
            'safety': config.w_safety,
        }
    }
    
    with open(output_dir / 'grpo_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=config.learning_rate
    )

    # Initialize frozen reference policy for KL term (strict GRPO baseline)
    try:
        ref_model = deepcopy(model.model)
        ref_model.eval()
        ref_model.to(model.model.device)
        for param in ref_model.parameters():
            param.requires_grad = False
        logger.info("Initialized frozen reference policy for KL computation")
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize frozen reference policy. "
            "A valid reference policy is required for GRPO KL computation."
        ) from e
    
    # Training loop
    logger.info("Starting GRPO training loop...")
    logger.info("")

    training_stats = {
        'iterations': [],
        'rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'kl_divergences': [],
        'semantic_rewards': [],
        'metacognitive_rewards': [],
        'empathy_rewards': [],
        'proactivity_rewards': [],
        'safety_rewards': [],
    }

    # Resume support: load previous stats if available
    latest_iteration = _find_latest_grpo_iteration(output_dir)
    stats_path = output_dir / 'training_stats.json'
    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                previous_stats = json.load(f)
            if isinstance(previous_stats, dict) and previous_stats.get('iterations'):
                for key in training_stats.keys():
                    if key in previous_stats and isinstance(previous_stats[key], list):
                        training_stats[key] = previous_stats[key]
                latest_iteration = max(latest_iteration, int(training_stats['iterations'][-1]))
                logger.info(f"Resuming GRPO stats from iteration {latest_iteration}")
        except Exception as e:
            logger.warning(f"Could not load existing GRPO stats, starting fresh stats: {e}")

    start_iteration = latest_iteration
    if start_iteration >= config.num_iterations:
        logger.info(
            f"Requested num_iterations={config.num_iterations} already reached "
            f"(latest={start_iteration}). Nothing to run."
        )
        _save_grpo_artifacts(output_dir, training_stats)
        return

    # Use a persistent dataloader iterator and only recreate on exhaustion
    data_iter = iter(train_dataloader)
    
    for iteration in range(start_iteration, config.num_iterations):
        logger.info(f"Iteration {iteration + 1}/{config.num_iterations}")
        
        # Sample batch of prompts
        try:
            batch = next(data_iter)
            # With batch_size=1, batch['prompt'] should be a single string
            prompts = [batch['prompt']] if isinstance(batch['prompt'], str) else batch['prompt']
        except StopIteration:
            logger.warning("DataLoader exhausted, restarting...")
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
            prompts = [batch['prompt']] if isinstance(batch['prompt'], str) else batch['prompt']

        if max_prompts_per_batch > 0 and len(prompts) > max_prompts_per_batch:
            prompts = prompts[:max_prompts_per_batch]
        
        # Generate K responses per prompt
        logger.info(f"  Generating {config.num_generations_per_prompt} responses per prompt...")
        
        all_responses = []
        all_rewards = []
        all_component_rewards = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Prompts", leave=False):
                prompt_responses = []
                prompt_rewards = []
                
                for k in range(config.num_generations_per_prompt):
                    # Generate response
                    try:
                        response = model.generate(
                            prompt,
                            max_length=config.generation_max_length,
                            temperature=config.generation_temperature,
                            top_p=config.generation_top_p,
                            do_sample=True
                        )
                    except Exception as e:
                        debug_info = {
                            "iteration": iteration + 1,
                            "prompt_index": p_idx,
                            "generation_index": k,
                            "generation_max_length": config.generation_max_length,
                            "temperature": config.generation_temperature,
                            "top_p": config.generation_top_p,
                            "prompt_preview": str(prompt)[:240],
                        }

                        if hasattr(model, "_get_prompt_debug_info"):
                            try:
                                debug_info.update(
                                    model._get_prompt_debug_info(
                                        str(prompt),
                                        max(2, int(config.generation_max_length) - 1)
                                    )
                                )
                            except Exception as dbg_exc:
                                debug_info["prompt_debug_error"] = str(dbg_exc)

                        logger.error("GRPO generation failure context: %s", json.dumps(debug_info, default=str))
                        raise RuntimeError(
                            "GRPO generation failed. See prior log line for prompt/tokenizer/model diagnostics."
                        ) from e
                    
                    prompt_responses.append(response)
                    
                    # Compute reward
                    components = reward_engine.compute_all(
                        prompt=prompt,
                        response=response
                    )
                    reward = (
                        reward_engine.w_semantic * components['semantic'] +
                        reward_engine.w_metacognitive * components['metacognitive'] +
                        reward_engine.w_empathy * components['empathy'] +
                        reward_engine.w_proactivity * components['proactivity'] +
                        reward_engine.w_safety * components['safety']
                    )
                    
                    prompt_rewards.append(reward)
                    all_component_rewards.append(components)
                
                all_responses.append(prompt_responses)
                all_rewards.append(prompt_rewards)
        
        # Convert to tensors
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=model.model.device)
        
        # Compute group-relative advantages
        advantages = compute_group_relative_advantages(
            rewards_tensor,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update 
        logger.info(f"  Updating policy...")
        
        model.model.train()
        total_policy_loss = 0.0
        total_kl_div = 0.0
        
        optimizer.zero_grad()
        
        # Process backpropagation per prompt group
        for p_idx, prompt in enumerate(prompts):
            adv_subset = advantages[p_idx]
            
            for r_idx, response in enumerate(all_responses[p_idx]):
                # Format full sequence
                full_text = prompt + response
                inputs = model.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.generation_max_length
                ).to(model.model.device)
                
                # Get response length to mask prompt gradients
                prompt_ids = model.tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
                prompt_len = len(prompt_ids)
                
                # Forward pass current policy
                outputs = model.model(**inputs)
                logits = outputs.logits
                
                # Get log probs of the generated tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                log_probs_dist = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                log_probs = torch.gather(log_probs_dist, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Calculate reference log probs from frozen reference policy
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)
                    ref_logits = ref_outputs.logits
                    ref_shift_logits = ref_logits[..., :-1, :].contiguous()
                    ref_log_probs_dist = torch.nn.functional.log_softmax(ref_shift_logits, dim=-1)
                    ref_log_probs = torch.gather(
                        ref_log_probs_dist,
                        2,
                        shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                
                # Only train on the response tokens (mask out prompt log probs)
                if log_probs.size(1) > prompt_len:
                    response_log_probs = log_probs[:, prompt_len-1:].sum(dim=1)
                    ref_response_log_probs = ref_log_probs[:, prompt_len-1:].sum(dim=1)
                    
                    # Compute PPO losses
                    p_loss = compute_ppo_loss(
                        response_log_probs,
                        ref_response_log_probs,
                        adv_subset[r_idx].unsqueeze(0),
                        clip_range=config.clip_range
                    )
                    
                    kl = compute_kl_divergence(
                        response_log_probs,
                        ref_response_log_probs
                    )
                    
                    p_loss.backward()
                    total_policy_loss += p_loss.item()
                    total_kl_div += kl.item()
        
        optimizer.step()
        model.model.eval()
        
        # Average the losses over all generations
        num_gens = len(prompts) * config.num_generations_per_prompt
        policy_loss = torch.tensor(total_policy_loss / max(num_gens, 1))
        kl_div = torch.tensor(total_kl_div / max(num_gens, 1))
        value_loss = torch.tensor(0.0) # Value loss mostly tracks critic in raw PPO
        
        # Log statistics
        avg_reward = rewards_tensor.mean().item()
        max_reward = rewards_tensor.max().item()
        min_reward = rewards_tensor.min().item()

        if all_component_rewards:
            avg_semantic = float(np.mean([x['semantic'] for x in all_component_rewards]))
            avg_metacognitive = float(np.mean([x['metacognitive'] for x in all_component_rewards]))
            avg_empathy = float(np.mean([x['empathy'] for x in all_component_rewards]))
            avg_proactivity = float(np.mean([x['proactivity'] for x in all_component_rewards]))
            avg_safety = float(np.mean([x['safety'] for x in all_component_rewards]))
        else:
            avg_semantic = avg_metacognitive = avg_empathy = avg_proactivity = avg_safety = 0.0
        
        training_stats['iterations'].append(iteration + 1)
        training_stats['rewards'].append(avg_reward)
        training_stats['policy_losses'].append(policy_loss.item())
        training_stats['value_losses'].append(value_loss.item())
        training_stats['kl_divergences'].append(kl_div.item())
        training_stats['semantic_rewards'].append(avg_semantic)
        training_stats['metacognitive_rewards'].append(avg_metacognitive)
        training_stats['empathy_rewards'].append(avg_empathy)
        training_stats['proactivity_rewards'].append(avg_proactivity)
        training_stats['safety_rewards'].append(avg_safety)
        
        logger.info(f"  Avg Reward: {avg_reward:.4f} (min: {min_reward:.4f}, max: {max_reward:.4f})")
        logger.info(
            f"  Policy Loss: {policy_loss.item():.8f} "
            f"({policy_loss.item():.3e})"
        )
        logger.info(
            f"  KL Divergence: {kl_div.item():.8f} "
            f"({kl_div.item():.3e})"
        )
        logger.info(
            "  Reward Components: "
            f"sem={avg_semantic:.4f}, "
            f"meta={avg_metacognitive:.4f}, "
            f"emp={avg_empathy:.4f}, "
            f"pro={avg_proactivity:.4f}, "
            f"safe={avg_safety:.4f}"
        )
        logger.info("")

        # Save running stats periodically for recoverability and live plotting
        if (iteration + 1) % max(config.eval_every_n_iterations, 10) == 0:
            _save_grpo_artifacts(output_dir, training_stats)
        
        # Save checkpoint periodically
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_path = output_dir / f"iteration_{iteration + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            model.model.save_pretrained(str(checkpoint_path))
            model.tokenizer.save_pretrained(str(checkpoint_path))
            
            logger.info(f"  ✓ Saved checkpoint to: {checkpoint_path}")
            logger.info("")
        
        # Evaluate periodically
        if (iteration + 1) % config.eval_every_n_iterations == 0:
            logger.info(f"  Running evaluation...")
            # Evaluation logic here
            logger.info("")
        
        # Early stopping based on KL divergence
        if kl_div.item() > config.target_kl * 1.5:
            logger.warning(f"  KL divergence {kl_div.item():.4f} exceeds target {config.target_kl}")
            logger.warning(f"  Consider reducing learning rate")
    
    # Save final model
    logger.info("Saving final policy...")
    final_path = output_dir / "final_policy"
    final_path.mkdir(parents=True, exist_ok=True)
    
    model.model.save_pretrained(str(final_path))
    model.tokenizer.save_pretrained(str(final_path))
    
    # Save training statistics/artifacts
    _save_grpo_artifacts(output_dir, training_stats)
    
    logger.info("")
    logger.info("="*80)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"  Final average reward: {training_stats['rewards'][-1]:.4f}")
    logger.info(f"  Total iterations: {config.num_iterations}")
    logger.info(f"  Model saved to: {final_path}")
    logger.info("="*80)
    logger.info("")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_group_relative_advantages(
    rewards: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> torch.Tensor:
    """
    Compute group-relative advantages using GAE.
    
    Args:
        rewards: Tensor of shape (batch_size, K) with rewards
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        Advantages tensor of same shape
    
    Example:
        >>> rewards = torch.tensor([[1.0, 2.0, 1.5], [0.8, 1.2, 1.0]])
        >>> advantages = compute_group_relative_advantages(rewards)
        >>> print(advantages.shape)
        torch.Size([2, 3])
    """
    # Group-relative baseline: mean reward within each group
    baselines = rewards.mean(dim=1, keepdim=True)
    
    # Advantages = rewards - baseline
    advantages = rewards - baselines
    
    return advantages


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO (Proximal Policy Optimization) loss.
    
    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Reference policy log probabilities
        advantages: Advantage estimates
        clip_range: Clipping parameter (ε)
    
    Returns:
        PPO loss (scalar)
    
    Example:
        >>> log_probs = torch.randn(32, 128)
        >>> old_log_probs = torch.randn(32, 128)
        >>> advantages = torch.randn(32, 128)
        >>> loss = compute_ppo_loss(log_probs, old_log_probs, advantages)
        >>> print(loss.shape)
        torch.Size([])
    """
    # Probability ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    
    # PPO loss (negative because we want to maximize)
    loss = -torch.min(surr1, surr2).mean()
    
    return loss


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between current and reference policy.
    
    Args:
        log_probs: Current policy log probabilities
        ref_log_probs: Reference policy log probabilities
    
    Returns:
        KL divergence (scalar)
    
    Example:
        >>> log_probs = torch.randn(32, 128)
        >>> ref_log_probs = torch.randn(32, 128)
        >>> kl = compute_kl_divergence(log_probs, ref_log_probs)
        >>> print(kl.shape)
        torch.Size([])
    """
    kl = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=-1).mean()
    return kl


def generate_k_responses(
    model,
    prompt: str,
    K: int,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> List[str]:
    """
    Generate K responses for a single prompt.
    
    Args:
        model: MedicalDigitalTwinModel
        prompt: Input prompt
        K: Number of responses to generate
        max_length: Maximum response length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        List of K generated responses
    
    Example:
        >>> model = MedicalDigitalTwinModel(config)
        >>> responses = generate_k_responses(model, "Patient with fever", K=5)
        >>> print(len(responses))
        5
    """
    responses = []
    
    for _ in range(K):
        response = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        responses.append(response)
    
    return responses