from dataclasses import dataclass, field

import torch
from transformers import AutoModel, AutoTokenizer

from pipeline.step_speech_features import SpeechTranscriber


@dataclass
class TextEncodeResult:
    subtitle: str
    embedding: torch.Tensor
    segments: list = field(default_factory=list)  # [(start_sec, end_sec, text), ...]


class TextEncoder:
    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        asr_model_path: str = "base",
        device: str = "cuda",
        local_files_only: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.transcriber = SpeechTranscriber(asr_model_path, device=device, batch_size=8)
        self.asr = self.transcriber.model

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=local_files_only)
        self.text_model = AutoModel.from_pretrained(text_model_name, local_files_only=local_files_only).to(self.device)
        self.text_model.eval()

    def audio_to_text(self, wav_path: str) -> tuple[str, list]:
        """ASR 转录，返回 (全文, [(start, end, text), ...])。"""
        try:
            result = self.transcriber.transcribe(wav_path)
            self.asr = self.transcriber.model
            return result.text, result.segments
        except Exception:
            return "", []

    def text_embedding(self, text: str) -> torch.Tensor:
        content = (text or "").strip()
        if not content:
            return torch.zeros(768, dtype=torch.float32)

        tokens = self.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.text_model(**tokens)
            cls_feat = out.last_hidden_state[:, 0, :].squeeze(0)

        return cls_feat.detach().cpu().float()

    def encode_fields(self, wav_path: str, title: str, tags: str) -> TextEncodeResult:
        subtitle, seg_list = self.audio_to_text(wav_path)
        merged = " ".join([x for x in [title or "", tags or "", subtitle or ""] if x]).strip()
        emb = self.text_embedding(merged)
        return TextEncodeResult(subtitle=subtitle, embedding=emb, segments=seg_list)
