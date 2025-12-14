"""
Vietnamese Phonemizer for TTS
"""

import os
import json
import glob
import platform
from typing import Optional, Dict

if platform.system() == "Windows":
    espeak_library_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    
    if os.path.exists(espeak_library_path):
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library_path
    else:
        espeak_library_path_x86 = r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"
        if os.path.exists(espeak_library_path_x86):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library_path_x86

from phonemizer import phonemize
from phonemizer.backend.espeak.espeak import EspeakWrapper

from .normalize_text import VietnameseTTSNormalizer



PHONEME_DICT_PATH = os.getenv(
    'PHONEME_DICT_PATH',
    os.path.join(os.path.dirname(__file__), "phoneme_dict.json")
)


def load_phoneme_dict(path: Optional[str] = None) -> Dict[str, str]:
    """
    Load phoneme dictionary from JSON file.
    
    Args:
        path: Path to the phoneme dictionary JSON file.
              Defaults to PHONEME_DICT_PATH.
    
    Returns:
        Dictionary mapping words to their phoneme representations.
    
    Raises:
        FileNotFoundError: If the dictionary file is not found
    """
    if path is None:
        path = PHONEME_DICT_PATH
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phoneme dictionary not found at {path}. "
            "Please create it or set PHONEME_DICT_PATH environment variable."
        )


def save_phoneme_dict(phoneme_dict: Dict[str, str], path: Optional[str] = None) -> None:
    """
    Save phoneme dictionary to JSON file.
    
    Args:
        phoneme_dict: Dictionary to save
        path: Path to save the dictionary. Defaults to PHONEME_DICT_PATH.
    """
    if path is None:
        path = PHONEME_DICT_PATH
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(phoneme_dict, f, ensure_ascii=False, indent=2)


_espeak_initialized = False
_phoneme_dict: Optional[Dict[str, str]] = None
_normalizer: Optional[VietnameseTTSNormalizer] = None


def _ensure_initialized():
    """Ensure all module components are initialized."""
    global _espeak_initialized, _phoneme_dict, _normalizer
    
    _espeak_initialized = True
    
    if _phoneme_dict is None:
        try:
            _phoneme_dict = load_phoneme_dict()
        except FileNotFoundError:
            print(f"Warning: Phoneme dictionary not found. Using empty dictionary.")
            _phoneme_dict = {}
    
    if _normalizer is None:
        _normalizer = VietnameseTTSNormalizer()


def phonemize_text(text: str, normalize: bool = True) -> str:
    """
    Convert text to phonemes using eSpeak phonemizer.
    
    Args:
        text: Input text to phonemize
        normalize: Whether to normalize text before phonemization
    
    Returns:
        Phoneme representation of the text
    """
    _ensure_initialized()
    
    if normalize and _normalizer is not None:
        text = _normalizer.normalize(text)
    
    return phonemize(
        text,
        language="vi",
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags"
    )


def phonemize_with_dict(
    text: str,
    phoneme_dict: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    update_dict: bool = True
) -> str:
    """
    Phonemize text with dictionary lookup.
    
    Uses cached dictionary lookups for known words and falls back to
    eSpeak phonemization for unknown words.
    
    Args:
        text: Input text to phonemize
        phoneme_dict: Custom phoneme dictionary. Uses module-level dict if None.
        normalize: Whether to normalize text before phonemization
        update_dict: Whether to update dictionary with new phonemizations
    
    Returns:
        Phoneme representation of the text
    """
    _ensure_initialized()
    
    if phoneme_dict is None:
        phoneme_dict = _phoneme_dict
    
    # Normalize text
    if normalize and _normalizer is not None:
        text = _normalizer.normalize(text)
    
    words = text.split()
    result = []
    
    for word in words:
        # Kiểm tra trong từ điển
        if word in phoneme_dict:
            phone_word = phoneme_dict[word]
        else:
            # Phonemize bằng eSpeak
            try:
                phone_word = phonemize(
                    word,
                    language='vi',
                    backend='espeak',
                    preserve_punctuation=True,
                    with_stress=True,
                    language_switch='remove-flags'
                )
                
                # Xử lý đặc biệt cho âm 'r' tiếng Việt
                if word.lower().startswith('r'):
                    phone_word = 'ɹ' + phone_word[1:]
                
                # Cập nhật từ điển
                if update_dict:
                    phoneme_dict[word] = phone_word
                
            except Exception as e:
                print(f"Warning: Could not phonemize '{word}': {e}")
                phone_word = word
        
        result.append(phone_word)
    
    return ' '.join(result)


def get_phoneme_dict() -> Dict[str, str]:
    """Get the current phoneme dictionary."""
    _ensure_initialized()
    return _phoneme_dict


def update_phoneme_dict(new_entries: Dict[str, str]) -> None:
    """
    Update the phoneme dictionary with new entries.
    
    Args:
        new_entries: Dictionary of new word-phoneme pairs to add
    """
    _ensure_initialized()
    _phoneme_dict.update(new_entries)


def batch_phonemize(
    texts: list[str],
    use_dict: bool = True,
    normalize: bool = True
) -> list[str]:
    """
    Phonemize a batch of texts.
    
    Args:
        texts: List of texts to phonemize
        use_dict: Whether to use dictionary lookup
        normalize: Whether to normalize texts
    
    Returns:
        List of phonemized texts
    """
    if use_dict:
        return [phonemize_with_dict(t, normalize=normalize) for t in texts]
    else:
        return [phonemize_text(t, normalize=normalize) for t in texts]


def create_phoneme_dict_from_texts(
    texts: list[str],
    normalize: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Create a phoneme dictionary from a list of texts.
    Useful for preprocessing training data.
    
    Args:
        texts: List of texts to process
        normalize: Whether to normalize texts
        save_path: Path to save the dictionary (optional)
    
    Returns:
        Dictionary mapping words to phonemes
    """
    _ensure_initialized()
    
    new_dict = {}
    
    for text in texts:
        if normalize and _normalizer is not None:
            text = _normalizer.normalize(text)
        
        words = text.split()
        for word in words:
            if word not in new_dict and word not in _phoneme_dict:
                try:
                    phone = phonemize(
                        word,
                        language='vi',
                        backend='espeak',
                        preserve_punctuation=True,
                        with_stress=True,
                        language_switch='remove-flags'
                    )
                    
                    if word.lower().startswith('r'):
                        phone = 'ɹ' + phone[1:]
                    
                    new_dict[word] = phone
                except Exception as e:
                    print(f"Warning: Could not phonemize '{word}': {e}")
    
    if save_path:
        # Merge with existing dictionary
        combined_dict = {**_phoneme_dict, **new_dict}
        save_phoneme_dict(combined_dict, save_path)
    
    return new_dict


if __name__ == "__main__":
    test_texts = [
        "Xin chào các bạn.",
        "Hôm nay là ngày 15/12/2025.",
        "Nhiệt độ là 25°C.",
        "Giá niêm yết là 1.500.000đ.",
    ]
    
    print("=" * 80)
    print("VIETNAMESE PHONEMIZATION TEST")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\n📝 Input: {text}")
        
        # Test phonemize_text
        phones1 = phonemize_text(text)
        print(f"🔤 phonemize_text: {phones1}")
        
        # Test phonemize_with_dict
        phones2 = phonemize_with_dict(text)
        print(f"📚 phonemize_with_dict: {phones2}")
        
        print("-" * 80)
