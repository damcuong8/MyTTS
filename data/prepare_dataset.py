import argparse
import os
from datasets import load_dataset, load_from_disk, Audio
from data.encode_audio import AudioEncoder
from utils.phonemize_text import phonemize_text, phonemize_with_dict, load_phoneme_dict

def prepare_dataset(
    dataset_path: str,
    output_dir: str,
    dataset_config: str = "default",
    split: str = "train",
    text_column: str = "text",
    audio_column: str = "audio",
    num_proc: int = 4,
    batch_size: int = 16,
    codec_repo: str = "neuphonic/neucodec",
    use_phoneme_dict: bool = True
):
    """
    Prepare dataset for training
    """
    
    print(f"Loading dataset: {dataset_path}...")
    try:
        if os.path.isdir(dataset_path):
            ds = load_from_disk(dataset_path)
        else:
            ds = load_dataset(dataset_path, dataset_config, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset info: {ds}")

    phoneme_dict = None
    if use_phoneme_dict:
        try:
            phoneme_dict = load_phoneme_dict()
            print("Loaded phoneme dictionary.")
        except:
            print("Phoneme dictionary not found, using pure espeak.")

    def process_phonemes(batch):
        texts = batch[text_column]
        phonemes_list = []
        for text in texts:
            if phoneme_dict:
                ph = phonemize_with_dict(text, phoneme_dict)
            else:
                ph = phonemize_text(text)
            phonemes_list.append(ph)
        return {"phonemes": phonemes_list}

    print("Running phonemization...")
    ds = ds.map(
        process_phonemes,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        desc="Phonemizing text"
    )

    ds = ds.cast_column(audio_column, Audio())
    
    encoder = AudioEncoder(codec_repo=codec_repo, target_sample_rate=16000)

    def process_audio_codes(batch):
        audio_items = batch[audio_column]
        codes_list = encoder.process_batch(audio_items)
        return {"codes": codes_list}

    print("Running audio encoding...")

    ds = ds.map(
        process_audio_codes,
        batched=True,
        batch_size=batch_size,
        desc="Encoding audio to codes"
    )

    ds = ds.filter(lambda x: x["codes"] is not None)

    print(f"Saving prepared dataset to {output_dir}...")
    ds.save_to_disk(output_dir)
    print("Done!")

if __name__ == "__main__":
    prepare_dataset(
        dataset_path="data/train_dataset",
        output_dir="data/train_dataset_encoded",
        codec_repo="neuphonic/neucodec"
    )
