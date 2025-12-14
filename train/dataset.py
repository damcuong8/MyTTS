
import json
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
from datasets import load_from_disk, load_dataset


class VieTTSDataset(Dataset):
    """
    Wrapper for HuggingFace datasets to work with our training pipeline.
    Supports both pre-processed datasets (with 'phonemes' and 'codes') 
    and raw datasets (requires on-the-fly processing).
    """
    
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        phonemizer: Optional[Callable[[str], str]] = None,
        max_seq_len: int = 2048,
        text_column: str = "text",
        codes_column: str = "codes",
        phonemes_column: str = "phonemes",
    ):
        """
        Initialize wrapper.
        
        Args:
            hf_dataset: HuggingFace dataset object
            tokenizer: HuggingFace tokenizer
            phonemizer: Phonemization function (optional if dataset has phonemes)
            max_seq_len: Maximum sequence length
            text_column: Column name for text
            codes_column: Column name for codes
            phonemes_column: Column name for phonemes
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.phonemizer = phonemizer
        self.max_seq_len = max_seq_len
        self.text_column = text_column
        self.codes_column = codes_column
        self.phonemes_column = phonemes_column
        
        self.speech_gen_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.ignore_index = -100
        
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        
        if self.phonemes_column in sample and sample[self.phonemes_column] is not None:
            phones = sample[self.phonemes_column]
        else:
            text = sample[self.text_column]
            if self.phonemizer is not None:
                phones = self.phonemizer(text)
            else:
                phones = text
        
        codes = sample[self.codes_column]
        if codes is None:
             codes = []
        
        codes_str = "".join([f"<|speech_{c}|>" for c in codes])
        
        # Create chat template
        chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
        
        # Tokenize
        ids = self.tokenizer.encode(chat)
        
        # Pad or truncate
        if len(ids) < self.max_seq_len:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(ids))
        else:
            ids = ids[:self.max_seq_len]
        
        input_ids = torch.tensor(ids, dtype=torch.long)
        
        # Create labels
        labels = torch.full_like(input_ids, self.ignore_index)
        speech_gen_start_positions = (input_ids == self.speech_gen_start_id).nonzero(as_tuple=True)[0]
        
        if len(speech_gen_start_positions) > 0:
            start_idx = speech_gen_start_positions[0].item()
            labels[start_idx:] = input_ids[start_idx:]
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def create_train_dataset(
    config,
    tokenizer,
    phonemizer: Optional[Callable[[str], str]] = None,
) -> Dataset:
    """
    Create training dataset from config using HF Datasets.
    
    Args:
        config: OmegaConf config object
        tokenizer: HuggingFace tokenizer
        phonemizer: Phonemization function
    
    Returns:
        Training dataset
    """
    dataset_config = config.get('dataset', {})
    
    if 'hf_dataset' not in dataset_config:
         raise ValueError("Config must contain 'hf_dataset' path (local or hub)")

    path = dataset_config['hf_dataset']
    logger.info(f"Loading HF dataset from: {path}")
    
    try:
        if Path(path).exists() and (Path(path) / "dataset_info.json").exists():
            hf_dataset = load_from_disk(path)
            logger.info("Loaded dataset from local disk.")
        else:
            hf_dataset = load_dataset(
                path,
                split=dataset_config.get('hf_split', 'train')
            )
            logger.info("Loaded dataset using load_dataset (Hub/Script).")
            
        return VieTTSDataset(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            phonemizer=phonemizer,
            max_seq_len=config.max_seq_len,
            text_column=dataset_config.get('text_column', 'text'),
            codes_column=dataset_config.get('codes_column', 'codes'),
            phonemes_column=dataset_config.get('phonemes_column', 'phonemes'),
        )
    except Exception as e:
        logger.error(f"Failed to load HF dataset: {e}")
        raise e
