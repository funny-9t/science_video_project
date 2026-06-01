import torch
import torch.nn.functional as F
from faster_whisper import WhisperModel


class AudioEncoder:
    def __init__(
        self,
        model_path: str = "base",
        device: str = "cuda",
        out_dim: int = 384,
        shared_model: WhisperModel | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.compute_type = "float16" if self.device.type == "cuda" else "int8"
        asr_device = "cuda" if self.device.type == "cuda" else "cpu"
        if shared_model is not None:
            self.model = shared_model
        else:
            try:
                self.model = WhisperModel(model_path, device=asr_device, compute_type=self.compute_type)
            except RuntimeError:
                self.model = WhisperModel(model_path, device="cpu", compute_type="int8")
        self.out_dim = int(out_dim)

    def encode(self, wav_path: str) -> torch.Tensor:
        try:
            enc = self.model.encode_audio(str(wav_path))  # [n_ctx, dim]
            if isinstance(enc, torch.Tensor):
                pooled = enc.mean(dim=0).float()
            else:
                pooled = torch.tensor(enc).mean(dim=0).float()
        except Exception:
            pooled = torch.zeros(1024, dtype=torch.float32)

        # deterministic mapping to fixed output dimension (384)
        mapped = F.adaptive_avg_pool1d(pooled.view(1, 1, -1), self.out_dim).view(-1)
        mapped = mapped.detach().cpu().float()
        if mapped.numel() != self.out_dim:
            raise RuntimeError(f"audio embedding dimension must be {self.out_dim}, got {mapped.numel()}")
        return mapped
