"""
MyVie-TTS Utilities Package
===========================

Các module tiện ích cho xử lý text tiếng Việt.
"""

from .normalize_text import VietnameseTTSNormalizer, split_text_into_chunks
from .phonemize_text import phonemize_text, phonemize_with_dict

__all__ = [
    'VietnameseTTSNormalizer',
    'split_text_into_chunks',
    'phonemize_text',
    'phonemize_with_dict',
]

