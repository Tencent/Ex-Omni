# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return None


class SpeechFeatureBatch(dict):
    """Minimal dict-like batch with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class HFSpeechEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        speech_encoder_type: str = "whisper",
        use_gradient_checkpointing: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = False,
    ):
        super().__init__()
        if speech_encoder_type.lower() != "whisper":
            raise ValueError("Open-source inference only supports speech_encoder_type='whisper'.")

        self.speech_encoder_type = "whisper"
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.model = self._load_model()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.hidden_size = getattr(self.model.config, "d_model", 1280)
        self.n_mels = getattr(self.feature_extractor, "feature_size", 80)
        self.input_stride = 2

        if use_gradient_checkpointing:
            if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "gradient_checkpointing_enable"):
                self.model.encoder.gradient_checkpointing_enable()
            elif hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        self.config = self.model.config

    def _load_model(self):
        return AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )

    @property
    def encoder(self):
        return getattr(self.model, "encoder", self.model)

    def forward(
        self,
        input_features: torch.Tensor,
        speech_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(input_features)
        return encoder_outputs.last_hidden_state

    def get_output_lengths(self, speech_lengths: torch.Tensor) -> torch.Tensor:
        speech_lengths = speech_lengths.view(-1)
        return ((speech_lengths + 1) // self.input_stride).clamp(min=1)

    def reload_pretrained_weights(self, model_name_or_path: Optional[str] = None):
        model_name_or_path = model_name_or_path or self.model_name_or_path
        self.model_name_or_path = model_name_or_path
        device = next(self.model.parameters()).device
        self.model = self._load_model().to(device=device)
        if self.torch_dtype is not None:
            self.model = self.model.to(dtype=self.torch_dtype)
        self.config = self.model.config

    @classmethod
    def load(cls, model_config):
        model_name_or_path = model_config.pretrain_speech_encoder_weights
        use_gradient_checkpointing = getattr(model_config, "use_gradient_checkpointing", False)
        torch_dtype = resolve_torch_dtype(getattr(model_config, "model_torch_dtype", None))
        low_cpu_mem_usage = getattr(model_config, "low_cpu_mem_usage", False)
        return cls(
            model_name_or_path=model_name_or_path,
            speech_encoder_type="whisper",
            use_gradient_checkpointing=use_gradient_checkpointing,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
