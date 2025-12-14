
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from loguru import logger


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback for TTS training."""
    
    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs
    ):
        if logs is not None and state.global_step % self.log_every_n_steps == 0:
            # Format logs for display
            log_str = f"Step {state.global_step}/{state.max_steps}"
            
            if 'loss' in logs:
                log_str += f" | Loss: {logs['loss']:.4f}"
            if 'learning_rate' in logs:
                log_str += f" | LR: {logs['learning_rate']:.2e}"
            if 'grad_norm' in logs:
                log_str += f" | Grad Norm: {logs['grad_norm']:.3f}"
            
            logger.info(log_str)


class CheckpointCallback(TrainerCallback):
    """Callback to manage checkpoints and save best model."""
    
    def __init__(
        self,
        keep_n_checkpoints: int = 3,
        save_best: bool = True,
        metric_for_best: str = "loss"
    ):
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.best_metric = float('inf')
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        # Clean up old checkpoints
        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1])
        )
        
        if len(checkpoints) > self.keep_n_checkpoints:
            for ckpt in checkpoints[:-self.keep_n_checkpoints]:
                logger.info(f"Removing old checkpoint: {ckpt}")
                import shutil
                shutil.rmtree(ckpt)
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float] = None,
        **kwargs
    ):
        if self.save_best and metrics is not None:
            current_metric = metrics.get(f"eval_{self.metric_for_best}", float('inf'))
            
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                logger.info(f"New best model! {self.metric_for_best}: {current_metric:.4f}")
                
                # Save best model
                best_path = Path(args.output_dir) / "best_model"
                kwargs['model'].save_pretrained(best_path)
                kwargs['tokenizer'].save_pretrained(best_path)
                logger.info(f"Saved best model to {best_path}")


class GradientMonitorCallback(TrainerCallback):
    """Monitor gradient statistics during training."""
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if state.global_step % self.log_every_n_steps == 0:
            model = kwargs.get('model')
            if model is not None:
                # Calculate gradient statistics
                total_norm = 0.0
                param_count = 0
                zero_grad_count = 0
                
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        param_count += 1
                        if p.grad.data.abs().max().item() == 0:
                            zero_grad_count += 1
                
                total_norm = total_norm ** 0.5
                
                if param_count > 0:
                    logger.debug(
                        f"Step {state.global_step} - "
                        f"Grad norm: {total_norm:.3f}, "
                        f"Zero grads: {zero_grad_count}/{param_count}"
                    )


def create_training_arguments(config) -> TrainingArguments:
    """
    Create TrainingArguments from config.
    
    Args:
        config: OmegaConf config object
    
    Returns:
        TrainingArguments instance
    """
    output_dir = os.path.join(config.save_root, config.run_name)
    
    # Determine precision
    bf16 = config.get('bf16', True)
    fp16 = config.get('fp16', False)
    
    # Determine report_to
    logging_config = config.get('logging', {})
    report_to = logging_config.get('report_to', 'tensorboard')
    
    # FSDP Configuration
    fsdp = config.get('fsdp', None)
    fsdp_config = config.get('fsdp_config', None)
    
    # Process fsdp_config: can be dict, path to JSON file, or None
    if fsdp_config is not None:
        if isinstance(fsdp_config, str):
            # Check if it's a path to JSON file
            if os.path.exists(fsdp_config) and fsdp_config.endswith('.json'):
                # It's a valid path to JSON file, pass as is
                logger.info(f"Using FSDP config from file: {fsdp_config}")
            else:
                # It's not a valid path, might be invalid config
                logger.warning(f"fsdp_config path '{fsdp_config}' does not exist. Ignoring FSDP config.")
                fsdp_config = None
        elif isinstance(fsdp_config, dict):
            # It's already a dict, pass as is
            logger.info("Using FSDP config from YAML dict")
        else:
            logger.warning(f"fsdp_config must be dict or path to JSON file, got {type(fsdp_config)}. Ignoring.")
            fsdp_config = None
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        do_train=True,
        learning_rate=config.lr,
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        
        # Batch size
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        
        # Precision
        bf16=bf16,
        fp16=fp16,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        
        # Logging
        logging_steps=config.logging_steps,
        report_to=report_to,
        
        # Evaluation
        eval_strategy="steps" if config.get('eval_steps') else "no",
        eval_steps=config.get('eval_steps'),
        
        # Data loading
        dataloader_num_workers=config.get('dataloader_num_workers', 4),
        dataloader_drop_last=config.get('dataloader_drop_last', True),
        remove_unused_columns=False,
        
        # Optimization
        torch_compile=config.get('torch_compile', True),
        gradient_checkpointing=config.get('gradient_checkpointing', False),
        
        # Distributed Training (FSDP)
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        
        # Reproducibility
        seed=config.seed,
        
        # Misc
        ignore_data_skip=True,
    )
    
    return args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    config=None,
    training_args: TrainingArguments = None,
    callbacks: List[TrainerCallback] = None,
) -> Trainer:
    """
    Create Trainer instance for TTS fine-tuning.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Config object (used if training_args is None)
        training_args: TrainingArguments (optional)
        callbacks: List of callbacks (optional)
    
    Returns:
        Trainer instance
    """
    # Create training arguments if not provided
    if training_args is None:
        if config is None:
            raise ValueError("Either config or training_args must be provided")
        training_args = create_training_arguments(config)
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
    
    # Add default callbacks
    callbacks.extend([
        LoggingCallback(log_every_n_steps=training_args.logging_steps),
        CheckpointCallback(keep_n_checkpoints=3),
    ])
    
    # Add early stopping if configured
    if config is not None:
        advanced_config = config.get('advanced', {})
        if advanced_config.get('early_stopping', False):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=advanced_config.get('early_stopping_patience', 5),
                    early_stopping_threshold=advanced_config.get('early_stopping_threshold', 0.0001),
                )
            )
    

    from transformers import default_data_collator
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    
    return trainer


def find_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output directory.
    
    Args:
        output_dir: Directory to search
    
    Returns:
        Path to latest checkpoint or None
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    checkpoints = list(output_path.glob("checkpoint-*"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.name.split("-")[1]),
        reverse=True
    )
    
    return str(checkpoints[0])


def resume_from_checkpoint(trainer: Trainer, output_dir: str) -> bool:
    """
    Resume training from checkpoint if available.
    
    Args:
        trainer: Trainer instance
        output_dir: Directory containing checkpoints
    
    Returns:
        True if resumed from checkpoint, False otherwise
    """
    checkpoint = find_checkpoint(output_dir)
    
    if checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
        return True
    
    return False
