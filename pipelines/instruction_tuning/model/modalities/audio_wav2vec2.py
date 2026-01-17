from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

from pipelines.instruction_tuning.model.data_tools import load_audio
from pipelines.instruction_tuning.model.modalities.base_modality import Modality
from pipelines.instruction_tuning.model.modalities.projectors import (
    build_patch_mlp_projector
)

class Wav2Vec2AudioModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.feature_extractor = None
        self.load_model()

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name_or_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        audios = audios.to(device=self.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=False):
            outputs = self.model(
                audios,
                output_hidden_states=True,
                return_dict=True
            )

        # (B, T', 768)
        hidden = outputs.hidden_states[-1]

        pooled = torch.nn.functional.adaptive_avg_pool1d(
            hidden.transpose(1, 2),
            output_size=300
        ).transpose(1, 2)  # (B, 300, 768)

        return pooled

class Wav2Vec2AudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "facebook/wav2vec2-base-960h",
        num_projector_layers: int = 2,
        num_tokens_output: int = 300,
        emb_size: int = 768
    ):
        self.model_name_or_path = model_name_or_path
        self.module = Wav2Vec2AudioModule(model_name_or_path)
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output
        self.emb_size = emb_size

    @property
    def name(self) -> str:
        return "audio_wav2vec2"

    @property
    def token(self) -> str:
        return "<speech>"

    @property
    def data_key(self) -> str:
        return "speech_audios"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_patch_mlp_projector(
            input_hidden_size=self.emb_size,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
        )

    def to(self, dtype: torch.dtype, device: torch.device) -> "Wav2Vec2AudioModality":
        self.module.to(device=device)
        self.module.model.to(device=device, dtype=torch.float32)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[torch.Tensor]]:
        row_values = []

        for row in rows:
            audios = []
            for audio_dict in row[self.data_key]:
                audio_dict = load_audio(
                    audio_dict,
                    target_sampling_rate=self.module.feature_extractor.sampling_rate,
                )

                audio_tensor = self.module.feature_extractor(
                    audio_dict["array"],
                    return_tensors="pt",
                    sampling_rate=audio_dict["sampling_rate"]
                ).input_values  # (1, T)

                audios.append(audio_tensor)

            row_values.append(
                torch.cat(audios, dim=0) if len(audios) > 0 else None
            )

        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []

        for audio_batch in encoded_values:
            feats = self.module(audio_batch)  # (B, 300, 768), fp32
            feats = feats.to(dtype=torch.bfloat16)

            outputs.append(feats)

        return outputs