import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Tuple

from ml.data_utils import normalize_text, chunk_text_by_tokens
from ml.summarizer import load_extractive_model, summarize_extractive


class VideoProcessingError(Exception):
    """Base error for video summarization failures."""


class AudioExtractionError(VideoProcessingError):
    """Raised when audio extraction fails."""


class TranscriptionFailedError(VideoProcessingError):
    """Raised when ASR output is invalid or low quality."""


class SummarizationError(VideoProcessingError):
    """Raised when summarization fails."""


class VideoSummarizer:
    """End-to-end video summarization pipeline."""

    _asr_processor = None
    _asr_model = None
    _asr_device = None
    _summarizer_models = {}
    _summarizer_tokenizers = {}
    _extractive_vectorizer = None
    _extractive_config = None

    def __init__(self, max_sentences: int | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.asr_model_name = os.getenv("ASR_MODEL", "facebook/wav2vec2-base-960h")
        self.summarizer_model_name = os.getenv(
            "SUMMARIZER_MODEL", "facebook/pegasus-cnn_dailymail"
        )
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.model_dir = os.getenv("MODEL_DIR", "outputs/cnn_dm_extractive")
        self.max_sentences = max_sentences or int(os.getenv("MAX_SUMMARY_SENTENCES", "5"))

    def summarize_video(self, video_path: str) -> Dict[str, object]:
        """Summarize a video file and return transcription, summary, and metadata."""
        start = time.time()
        result = self.transcribe_video(video_path)
        transcription = result.get("transcription", "")
        confidence = result.get("confidence", 0.0)
        duration = result.get("duration_seconds", 0.0)
        summary, summarizer_name = self.summarize_text(transcription, confidence)
        if not summary:
            raise SummarizationError("Summary generation failed")
        latency_ms = (time.time() - start) * 1000.0
        self.logger.info("Video summary generated in %.2fms", latency_ms)
        return {
            "transcription": transcription,
            "summary": summary,
            "duration_seconds": duration,
            "model_used": {
                "asr": self.asr_model_name,
                "summarizer": summarizer_name,
            },
        }

    def transcribe_video(self, video_path: str) -> Dict[str, object]:
        if not os.path.exists(video_path):
            raise VideoProcessingError(f"File not found: {video_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            duration = self.extract_audio(video_path, audio_path)
            transcription, confidence = self.transcribe_audio(audio_path)
        if not transcription:
            raise TranscriptionFailedError("Transcription is empty after cleaning")
        return {
            "transcription": transcription,
            "confidence": confidence,
            "duration_seconds": duration,
        }

    def extract_audio(self, video_path: str, output_path: str) -> float:
        """Extract normalized mono 16kHz PCM WAV audio from a video."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise AudioExtractionError("ffmpeg is not available on PATH")
        process = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                output_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0:
            error_details = process.stderr.decode("utf-8", errors="ignore").strip()
            raise AudioExtractionError(
                f"ffmpeg failed to extract audio: {error_details or 'unknown error'}"
            )
        duration = self._get_audio_duration(output_path)
        if duration <= 0:
            raise AudioExtractionError("Extracted audio has zero duration")
        return duration

    def transcribe_audio(self, audio_path: str) -> Tuple[str, float]:
        """Run ASR on a normalized WAV file and return text with confidence."""
        audio, sample_rate = self._load_audio(audio_path)
        audio = self._normalize_audio(audio)
        audio = self._trim_silence(audio, sample_rate)
        if audio.numel() == 0:
            raise TranscriptionFailedError("Audio contains only silence")
        chunks = self._chunk_audio(audio, sample_rate)
        if not chunks:
            raise TranscriptionFailedError("Audio is too short for transcription")
        processor, model, device = self._get_asr()
        transcript_parts = []
        confidences = []
        with self._no_grad():
            for chunk in chunks:
                inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = model(**inputs).logits
                predicted_ids = logits.argmax(dim=-1)
                transcript_parts.append(processor.batch_decode(predicted_ids)[0])
                confidences.append(self._logit_confidence(logits))
        transcription = " ".join(part.strip() for part in transcript_parts if part.strip())
        cleaned = self.clean_text(transcription)
        confidence = float(sum(confidences) / max(1, len(confidences)))
        if not self._is_transcription_valid(cleaned, confidence):
            raise TranscriptionFailedError("Transcription failed validation")
        return cleaned, confidence

    def clean_text(self, text: str) -> str:
        """Normalize ASR output for summarization."""
        normalized = normalize_text(text).lower()
        normalized = re.sub(r"\b(\[unk\]|\[pad\]|<s>|</s>)\b", " ", normalized)
        normalized = re.sub(r"(.)\1{3,}", r"\1\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return ""
        try:
            from nltk.tokenize import sent_tokenize

            sentences = [s.strip() for s in sent_tokenize(normalized) if s.strip()]
            if sentences:
                normalized = ". ".join(sentences)
        except Exception:
            pass
        return normalized

    def summarize_text(self, text: str, confidence: float) -> Tuple[str, str]:
        """Summarize cleaned text using extractive or abstractive models."""
        word_count = len(text.split())
        if confidence < 0.5 or word_count < 40:
            summary = self._summarize_extractive(self._prepare_extractive_text(text))
            return summary, "tfidf"
        try:
            summary = self._summarize_abstractive(text)
            if not summary:
                summary = self._summarize_extractive(self._prepare_extractive_text(text))
                return summary, "tfidf"
            return summary, self.summarizer_model_name
        except Exception as exc:
            self.logger.error("Abstractive summarization failed: %s", exc)
            summary = self._summarize_extractive(self._prepare_extractive_text(text))
            return summary, "tfidf"

    def _summarize_extractive(self, text: str) -> str:
        vectorizer = self._get_extractive_vectorizer()
        result = summarize_extractive(text=text, vectorizer=vectorizer, max_sentences=self.max_sentences)
        return result.get("summary", "").strip()

    def _summarize_abstractive(self, text: str) -> str:
        tokenizer, model, device = self._get_abstractive()
        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=512, stride=64)
        summaries = []
        with self._no_grad():
            for chunk in chunks:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                output_ids = model.generate(
                    **inputs,
                    num_beams=4,
                    max_length=128,
                    min_length=24,
                    do_sample=False,
                )
                summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        return normalize_text(" ".join(summaries))

    def _prepare_extractive_text(self, text: str) -> str:
        words = text.split()
        if len(words) < 80:
            return text
        try:
            from nltk.tokenize import sent_tokenize
            sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            sentences = []
        if len(sentences) >= 2:
            avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
            if avg_len <= 50:
                return text
        chunk_size = 25
        chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return ". ".join(chunks)

    def _get_extractive_vectorizer(self):
        if self.__class__._extractive_vectorizer is None:
            vectorizer, config = load_extractive_model(self.model_dir)
            self.__class__._extractive_vectorizer = vectorizer
            self.__class__._extractive_config = config
        return self.__class__._extractive_vectorizer

    def _get_abstractive(self):
        if self.summarizer_model_name in self.__class__._summarizer_models:
            model = self.__class__._summarizer_models[self.summarizer_model_name]
            tokenizer = self.__class__._summarizer_tokenizers[self.summarizer_model_name]
            return tokenizer, model, model.device
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise SummarizationError("transformers is not installed") from exc
        device = self._get_device()
        tokenizer = AutoTokenizer.from_pretrained(self.summarizer_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.summarizer_model_name)
        model = model.to(device)
        model.eval()
        self.__class__._summarizer_models[self.summarizer_model_name] = model
        self.__class__._summarizer_tokenizers[self.summarizer_model_name] = tokenizer
        return tokenizer, model, device

    def _get_asr(self):
        if self.__class__._asr_model is not None:
            return self.__class__._asr_processor, self.__class__._asr_model, self.__class__._asr_device
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise TranscriptionFailedError("transformers is not installed") from exc
        device = self._get_device()
        processor = Wav2Vec2Processor.from_pretrained(self.asr_model_name)
        model = Wav2Vec2ForCTC.from_pretrained(self.asr_model_name)
        model = model.to(device)
        model.eval()
        self.__class__._asr_processor = processor
        self.__class__._asr_model = model
        self.__class__._asr_device = device
        return processor, model, device

    def _get_device(self):
        try:
            import torch
        except ImportError as exc:
            raise VideoProcessingError("torch is not installed") from exc
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_audio(self, audio_path: str):
        try:
            import torchaudio
        except ImportError as exc:
            raise AudioExtractionError("torchaudio is not installed") from exc
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform is None or waveform.numel() == 0:
            raise TranscriptionFailedError("Audio file is empty")
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0).float()
        return waveform, sample_rate

    def _normalize_audio(self, audio):
        try:
            import torch
        except ImportError as exc:
            raise VideoProcessingError("torch is not installed") from exc
        max_val = torch.max(torch.abs(audio)).item()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def _trim_silence(self, audio, sample_rate: int):
        try:
            import torch
        except ImportError as exc:
            raise VideoProcessingError("torch is not installed") from exc
        threshold = float(os.getenv("ASR_SILENCE_THRESHOLD", "0.01"))
        mask = torch.abs(audio) > threshold
        if not torch.any(mask):
            return audio[:0]
        indices = torch.where(mask)[0]
        start = int(indices[0].item())
        end = int(indices[-1].item()) + 1
        return audio[start:end]

    def _chunk_audio(self, audio, sample_rate: int):
        chunk_seconds = float(os.getenv("ASR_CHUNK_SECONDS", "20"))
        overlap_seconds = float(os.getenv("ASR_CHUNK_OVERLAP_SECONDS", "2"))
        chunk_size = max(1, int(sample_rate * chunk_seconds))
        overlap = max(0, int(sample_rate * overlap_seconds))
        step = max(1, chunk_size - overlap)
        min_size = int(sample_rate * 0.5)
        chunks = []
        total = int(audio.numel())
        start = 0
        while start < total:
            end = min(total, start + chunk_size)
            chunk = audio[start:end]
            if chunk.numel() >= min_size:
                chunks.append(chunk)
            if end == total:
                break
            start += step
        return chunks

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            import torchaudio
        except ImportError as exc:
            raise AudioExtractionError("torchaudio is not installed") from exc
        info = torchaudio.info(audio_path)
        if info.sample_rate == 0:
            return 0.0
        return float(info.num_frames) / float(info.sample_rate)

    def _is_transcription_valid(self, text: str, confidence: float) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return False
        words = cleaned.split()
        if len(words) < 3:
            return False
        if len(words) < 5 and confidence < 0.7:
            return False
        if re.search(r"(.)\1{6,}", cleaned, re.IGNORECASE):
            return False
        if re.search(r"([aeiou])\1{4,}", cleaned, re.IGNORECASE):
            return False
        stripped = cleaned.replace(" ", "")
        non_alpha = sum(1 for ch in stripped if not ch.isalpha())
        return (non_alpha / len(stripped)) <= 0.5

    def _logit_confidence(self, logits) -> float:
        try:
            import torch
        except ImportError:
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        return float(max_probs.mean().item())

    def _no_grad(self):
        try:
            import torch
        except ImportError as exc:
            raise VideoProcessingError("torch is not installed") from exc
        return torch.no_grad()
