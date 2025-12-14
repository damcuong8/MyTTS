
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
    default_data_collator,
)

from train.dataset import create_train_dataset
from train.trainer import create_trainer

def setup_tokenizer(restore_from, codebook_size):
    tokenizer = AutoTokenizer.from_pretrained(restore_from)
    return tokenizer

def setup_model(restore_from: str, config):
    logger.info(f"Loading model from {restore_from}")
    
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
    
    if advanced.get('freeze_layers', 0) > 0:
        freeze_layers = advanced['freeze_layers']
        logger.info(f"Freezing first {freeze_layers} transformer layers")
        
        for i, layer in enumerate(model.model.layers[:freeze_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    if advanced.get('freeze_embeddings', False):
        logger.info("Freezing embedding layers")
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
    
    # Enable gradient checkpointing if specified
    if config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    return model


def setup_phonemizer(config):
    vietnamese_config = config.get('vietnamese', {})
    
    if not vietnamese_config.get('use_phoneme_dict', True):
        # Use basic phonemizer
        from utils.phonemize_text import phonemize_text
        logger.info("Using basic eSpeak phonemizer")
        return phonemize_text
    
    # Use dictionary-based phonemizer
    from utils.phonemize_text import phonemize_with_dict, load_phoneme_dict
    
    dict_path = vietnamese_config.get('phoneme_dict_path', './utils/phoneme_dict.json')
    
    try:
        phoneme_dict = load_phoneme_dict(dict_path)
        logger.info(f"Loaded phoneme dictionary with {len(phoneme_dict)} entries")
    except FileNotFoundError:
        logger.warning(f"Phoneme dictionary not found at {dict_path}. Using empty dictionary.")
        phoneme_dict = {}
    
    def phonemizer_fn(text):
        return phonemize_with_dict(text, phoneme_dict=phoneme_dict, normalize=True)
    
    return phonemizer_fn


def main(config_fpath: str):
    """
    Main FULL FINE-TUNING function.
    """
    logger.info(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)
    
    # Create output directory
    output_dir = os.path.join(config.save_root, config.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    
    logger.info("=" * 60)
    logger.info("MyVie-TTS FULL FINE-TUNING")
    logger.info("=" * 60)
    
    restore_from = config.restore_from
    

    tokenizer = AutoTokenizer.from_pretrained(restore_from)

    model = setup_model(
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
    
    # Setup Trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        config=config,
    )
    
    # Train
    logger.info("Starting training...")
    
    # Check for existing checkpoint
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
    
    # Save Final Model
    logger.info("Saving final model...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    Fire(main)
