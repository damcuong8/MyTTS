"""
Encode audio to NeuCodec tokens
"""
import torch
import librosa
import numpy as np
import torchaudio
from neucodec import NeuCodec, DistillNeuCodec
from typing import Optional, List, Union

DEFAULT_SAMPLE_RATE = 16000  # NeuCodec requires 16kHz

class AudioEncoder:
    """
    Audio encoder using NeuCodec
    """
    
    def __init__(
        self,
        codec_repo: str = "neuphonic/neucodec",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_distill: bool = False,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self._resampler_cache = {}
        
        print(f"Loading codec from {codec_repo} on {device}...")
        if use_distill or "distill" in codec_repo:
            self.codec = DistillNeuCodec.from_pretrained(codec_repo)
        else:
            self.codec = NeuCodec.from_pretrained(codec_repo)
            
        self.codec.eval().to(device)
        print(f"Codec loaded successfully! Target SR: {target_sample_rate}Hz (NeuCodec requirement)")

    def process_batch(self, batch_audio):
        """
        Process a batch of audio items from HF dataset.
        Expected input format from HF 'audio' feature:
        {
            'array': np.ndarray,
            'sampling_rate': int
        }
        """
        codes_list = []
        
        for audio_item in batch_audio:
            if audio_item is None:
                codes_list.append(None)
                continue

            wav = audio_item['array']
            sr = audio_item['sampling_rate']

            wav_tensor = torch.from_numpy(wav).float()

            if sr != self.target_sample_rate:

                if sr not in self._resampler_cache:
                    self._resampler_cache[sr] = torchaudio.transforms.Resample(
                        orig_freq=sr, 
                        new_freq=self.target_sample_rate
                    )
                resampler = self._resampler_cache[sr]
                wav_tensor = resampler(wav_tensor)
            

            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
            elif wav_tensor.dim() == 2:
                wav_tensor = wav_tensor.unsqueeze(0)
                
            wav_tensor = wav_tensor.to(self.device)
            

            with torch.no_grad():
                try:
                    # NeuCodec encode_code expects [B, C, T]
                    codes = self.codec.encode_code(audio_or_path=wav_tensor)
                    # codes shape usually [1, N_q, T']
                    codes = codes.squeeze(0).squeeze(0).cpu().numpy().tolist()
                    codes_list.append(codes)
                except Exception as e:
                    print(f"Error encoding audio: {e}")
                    codes_list.append(None)
                    
        return codes_list
