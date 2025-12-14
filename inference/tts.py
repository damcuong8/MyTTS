"""
MyVie-TTS Inference Module
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, List, Generator
import re

import numpy as np
import torch
import librosa
import soundfile as sf


from neucodec import NeuCodec, DistillNeuCodec
from transformers import AutoTokenizer, AutoModelForCausalLM


def _linear_overlap_add(frames: List[np.ndarray], stride: int) -> np.ndarray:
    """
    Perform linear overlap-add for streaming audio synthesis.
    
    Original implementation from:
    https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
    """
    assert len(frames), "frames list cannot be empty"
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    
    assert sum_weight.min() > 0
    return out / sum_weight


class MyVieTTS:
    """
    Vietnamese Text-to-Speech inference class.
    
    Usage:
        >>> tts = MyVieTTS(tts_model="path/to/model", device="cuda")
        >>> ref_codes = tts.encode_reference("reference.wav")
        >>> wav = tts.infer("Xin chào Việt Nam", ref_codes, "Reference text")
        >>> tts.save_audio(wav, "output.wav")
    """
    
    def __init__(
        self,
        tts_model: str = "model",
        device: str = "cpu",
        codec_repo: str = "neuphonic/neucodec",
        codec_device: str = "cpu",
        use_phoneme_dict: bool = True,
    ):
        """
        Initialize MyVieTTS.
        
        Args:
            tts_model: HuggingFace repo or local path for the TTS model
            device: Device for backbone model ('cpu', 'cuda', 'cuda:0', etc.)
            codec_repo: HuggingFace repo for NeuCodec
            codec_device: Device for codec model
            use_phoneme_dict: Whether to use phoneme dictionary for Vietnamese
        """
        # Constants
        self.sample_rate = 24_000
        self.max_context = 2048
        self.hop_length = 480
        
        # Streaming parameters
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length
        
        # Model flags
        self._is_onnx_codec = False
        
        # References
        self.tokenizer = None
        self.backbone = None
        self.codec = None
        
        # Load phonemizer
        self._setup_phonemizer(use_phoneme_dict)
        
        # Load models
        self._load_model(tts_model, device)
        self._load_codec(codec_repo, codec_device)
    
    def _setup_phonemizer(self, use_phoneme_dict: bool):
        """Setup phonemizer for Vietnamese."""
        try:
            from utils.phonemize_text import (
                phonemize_text, 
                phonemize_with_dict,
                load_phoneme_dict
            )
            
            if use_phoneme_dict:
                try:
                    self._phoneme_dict = load_phoneme_dict()
                    self._phonemizer = lambda text: phonemize_with_dict(
                        text, 
                        self._phoneme_dict,
                        normalize=True
                    )
                    print("Using phoneme dictionary for phonemization")
                except FileNotFoundError:
                    print("Phoneme dictionary not found, using basic phonemizer")
                    self._phonemizer = phonemize_text
            else:
                self._phonemizer = phonemize_text
                
        except Exception as e:
            print(f"Warning: Failed to setup Vietnamese phonemizer: {e}")
            print("Falling back to basic text (no phonemization)")
            self._phonemizer = lambda x: x
    
    def _load_model(self, tts_model: str, device: str):
        """Load backbone TTS model."""
        print(f"Loading backbone from: {tts_model} on {device}...")
        
        # Load HuggingFace model
        self.tokenizer = AutoTokenizer.from_pretrained(tts_model)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            tts_model,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        ).to(torch.device(device))
        self.backbone.eval()
        
        print("Backbone loaded successfully!")
    
    def _load_codec(self, codec_repo: str, codec_device: str):
        """Load NeuCodec vocoder."""
        print(f"Loading codec from: {codec_repo} on {codec_device}...")
        
        if codec_repo == "neuphonic/neucodec":
            self.codec = NeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)
        elif codec_repo == "neuphonic/distill-neucodec":
            self.codec = DistillNeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)
        elif codec_repo == "neuphonic/neucodec-onnx-decoder":
            if codec_device != "cpu":
                raise ValueError("ONNX decoder only currently runs on CPU.")
            
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import the ONNX decoder. "
                    "Ensure you have onnxruntime installed as well as neucodec >= 0.0.4."
                ) from e
            
            self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
            self._is_onnx_codec = True
        else:
            raise ValueError(
                f"Invalid codec repo: {codec_repo}. Must be one of: "
                "'neuphonic/neucodec', 'neuphonic/distill-neucodec', "
                "'neuphonic/neucodec-onnx-decoder'."
            )
        
        print("Codec loaded successfully!")
    
    def encode_reference(self, ref_audio_path: Union[str, Path]) -> np.ndarray:
        """
        Encode reference audio to codec tokens.
        
        Args:
            ref_audio_path: Path to reference audio file
        
        Returns:
            Numpy array of codec tokens
        """
        wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor)
            ref_codes = ref_codes.squeeze(0).squeeze(0)
            
            if isinstance(ref_codes, torch.Tensor):
                ref_codes = ref_codes.cpu().numpy()
        
        return ref_codes
    
    def _to_phones(self, text: str) -> str:
        """Convert text to phonemes."""
        return self._phonemizer(text)
    
    def _decode(self, codes_str: str) -> np.ndarray:
        """
        Decode speech tokens to audio waveform.
        
        Args:
            codes_str: String of speech tokens like "<|speech_123|><|speech_456|>..."
        
        Returns:
            Audio waveform as numpy array
        """
        # Extract token IDs
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]
        
        if len(speech_ids) == 0:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "The model may not have generated proper speech tokens."
            )
        
        # Decode with codec
        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :]
                codes = codes.to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()
        
        return recon[0, 0, :]
    
    def _apply_chat_template(
        self,
        ref_codes: np.ndarray,
        ref_text: str,
        input_text: str
    ) -> List[int]:
        """Create input token IDs from chat template."""
        # Phonemize texts
        input_phones = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        
        # Get special token IDs
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        
        # Encode input text
        input_ids = self.tokenizer.encode(input_phones, add_special_tokens=False)
        
        # Create chat template
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)
        
        # Replace text placeholder
        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1:]
        )
        
        # Replace speech placeholder with reference codes
        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        
        return ids
    
    def _infer_torch(self, prompt_ids: List[int]) -> str:
        """Inference using PyTorch model."""
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                min_new_tokens=50,
            )
        
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(),
            add_special_tokens=False
        )
        
        return output_str
    
    def _infer_torch_stream(
        self,
        prompt_ids: List[int],
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[str, None, None]:
        """Inference using PyTorch model with streaming."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )
        
        generation_kwargs = {
            "input_ids": prompt_tensor,
            "max_length": self.max_context,
            "eos_token_id": speech_end_id,
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "use_cache": True,
            "min_new_tokens": 50,
            "streamer": streamer,
        }
        
        thread = Thread(target=self.backbone.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    def infer(
        self,
        text: str,
        ref_codes: np.ndarray,
        ref_text: str,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Input text to convert to speech
            ref_codes: Encoded reference audio codes
            ref_text: Transcript of reference audio
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            Generated audio waveform as numpy array
        """
        # Generate tokens
        prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
        output_str = self._infer_torch(prompt_ids)
        
        # Decode to audio
        wav = self._decode(output_str)
        
        return wav
    
    def infer_stream(
        self,
        text: str,
        ref_codes: np.ndarray,
        ref_text: str,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech from text with streaming output.
        
        Args:
            text: Input text to convert to speech
            ref_codes: Encoded reference audio codes
            ref_text: Transcript of reference audio
        
        Yields:
            Audio chunks as numpy arrays
        """
        # Generate tokens
        prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
        
        audio_cache: List[np.ndarray] = []
        token_cache: List[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_samples: int = 0
        n_decoded_tokens: int = len(ref_codes)
        
        for output_str in self._infer_torch_stream(prompt_ids):
            token_cache.append(output_str)
            
            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                # Decode chunk
                tokens_start = max(
                    n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames,
                    0
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)
                
                # Postprocess with overlap-add
                processed_recon = _linear_overlap_add(
                    audio_cache, stride=self.streaming_stride_samples
                )
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                
                yield processed_recon
        
        # Final chunk
        remaining_tokens = len(token_cache) - n_decoded_tokens
        if len(token_cache) > n_decoded_tokens:
            tokens_start = max(
                len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens),
                0
            )
            sample_start = (
                len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames
            ) * self.hop_length
            
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = recon[sample_start:]
            audio_cache.append(recon)
            
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            
            yield processed_recon
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Path to save audio
            sample_rate: Sample rate (defaults to self.sample_rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")
    
    @property
    def device(self) -> str:
        """Get the device of the backbone model."""
        return str(self.backbone.device)


def quick_tts(
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str = "output.wav",
    model_path: str = "model",
    device: str = "cpu",
) -> np.ndarray:
    """
    Quick function for one-shot TTS.
    
    Args:
        text: Text to synthesize
        ref_audio: Path to reference audio
        ref_text: Transcript of reference audio
        output_path: Path to save output
        model_path: Path to model
        device: Device to use
    
    Returns:
        Generated audio waveform
    """
    tts = MyVieTTS(
        tts_model=model_path,
        device=device,
        codec_device=device,
    )
    
    ref_codes = tts.encode_reference(ref_audio)
    wav = tts.infer(text, ref_codes, ref_text)
    tts.save_audio(wav, output_path)
    
    return wav


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MyVieTTS Inference Module")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--ref_audio", type=str, help="Path to reference audio file")
    parser.add_argument("--ref_text", type=str, help="Transcript of reference audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--model", type=str, default="model", help="Path to TTS model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    if args.text and args.ref_audio and args.ref_text:
        # Run inference
        print("Starting TTS inference...")
        wav = quick_tts(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            output_path=args.output,
            model_path=args.model,
            device=args.device,
        )
        print(f"Inference completed! Audio saved to {args.output}")
    else:
        # Show usage
        print("MyVieTTS Inference Module")
        print("\nUsage:")
        print("  python tts.py --text 'Xin chào' --ref_audio ref.wav --ref_text 'Xin chào' --output out.wav")
        print("\nOr import as module:")
        print("  from inference.tts import MyVieTTS")
