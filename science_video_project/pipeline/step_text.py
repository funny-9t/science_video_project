from dataclasses import dataclass

import torch
from faster_whisper import WhisperModel
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextEncodeResult:
    subtitle: str
    embedding: torch.Tensor


class TextEncoder:
    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        asr_model_path: str = "base",
        device: str = "cuda",
        local_files_only: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.compute_type = "float16" if self.device.type == "cuda" else "int8"
        asr_device = "cuda" if self.device.type == "cuda" else "cpu"
        try:
            self.asr = WhisperModel(asr_model_path, device=asr_device, compute_type=self.compute_type)
        except RuntimeError:
            self.asr = WhisperModel(asr_model_path, device="cpu", compute_type="int8")

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, local_files_only=local_files_only)
        self.text_model = AutoModel.from_pretrained(text_model_name, local_files_only=local_files_only).to(self.device)
        self.text_model.eval()

    def audio_to_text(self, wav_path: str) -> str:
        try:
            segments, _ = self.asr.transcribe(wav_path)
            text = " ".join(seg.text.strip() for seg in segments if seg.text)
            return text.strip()
        except Exception:
            return ""

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
        subtitle = self.audio_to_text(wav_path)
        merged = " ".join([x for x in [title or "", tags or "", subtitle or ""] if x]).strip()
        emb = self.text_embedding(merged)
        return TextEncodeResult(subtitle=subtitle, embedding=emb)
