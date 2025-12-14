
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from fire import Fire
from omegaconf import OmegaConf
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from train.dataset import create_train_dataset
from train.trainer import create_trainer

def setup_model_lora(restore_from: str, config):
    logger.info(f"Loading base model from {restore_from}")
    
    if config.get('bf16', True):
        dtype = torch.bfloat16
    elif config.get('fp16', False):
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        restore_from,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    
    advanced = config.get('advanced', {})
    
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=advanced.get('lora_r', 16),
        lora_alpha=advanced.get('lora_alpha', 32),
        lora_dropout=advanced.get('lora_dropout', 0.05),
        target_modules=advanced.get('lora_target_modules', ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]),
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    model.print_trainable_parameters()
    
    return model

def setup_phonemizer(config):
    vietnamese_config = config.get('vietnamese', {})
    
    if not vietnamese_config.get('use_phoneme_dict', True):
        from utils.phonemize_text import phonemize_text
        return phonemize_text
    
    from utils.phonemize_text import phonemize_with_dict, load_phoneme_dict
    dict_path = vietnamese_config.get('phoneme_dict_path', './utils/phoneme_dict.json')
    
    try:
        phoneme_dict = load_phoneme_dict(dict_path)
    except FileNotFoundError:
        phoneme_dict = {}
    
    def phonemizer_fn(text):
        return phonemize_with_dict(text, phoneme_dict=phoneme_dict, normalize=True)
    
    return phonemizer_fn

def main(config_fpath: str):
    """
    Main function for LoRA fine-tuning.
    """
    logger.info(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    
    output_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config_save_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    
    logger.info("=" * 60)
    logger.info("MyVie-TTS LoRA FINE-TUNING")
    logger.info("=" * 60)
    
    restore_from = config.restore_from
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    
    model = setup_model_lora(
        restore_from=restore_from,
        config=config
    )
    
    phonemizer = setup_phonemizer(config)
    
    logger.info("Loading training dataset...")
    train_dataset = create_train_dataset(
        config=config,
        tokenizer=tokenizer,
        phonemizer=phonemizer
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        config=config,
    )
    
    logger.info("Starting training...")
    
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(
                output_dir,
                sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            )
            logger.info(f"Found checkpoint: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    logger.info("Saving final LoRA adapter...")
    final_model_path = os.path.join(output_dir, "final_adapter")
    
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"Final LoRA adapter saved to {final_model_path}")

if __name__ == "__main__":
    Fire(main)

