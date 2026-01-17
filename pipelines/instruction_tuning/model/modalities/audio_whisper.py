from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WhisperModel

from pipelines.instruction_tuning.model.data_tools import load_audio
from pipelines.instruction_tuning.model.modalities.base_modality import Modality
from pipelines.instruction_tuning.model.modalities.projectors import (
    build_mlp_vector_projector,
    build_patch_mlp_projector
)


OUTPUT_EMB_SIZE = 768 # whisper small
# OUTPUT_EMB_SIZE = 1024 # whisper medium
# OUTPUT_EMB_SIZE = 1280 # whisper large

class WhisperAudioModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.feature_extractor = None

        self.load_model()

    def load_model(self):
        self.model = WhisperModel.from_pretrained(self.model_name_or_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, audios) -> torch.Tensor:
        hidden_states = []
        
        for i in range(audios.shape[0]):  # Process each audio in the batch
            decoder_input_ids = (
                torch.tensor([[1]]) * self.model.config.decoder_start_token_id
            )
            
            model_output = self.model(
                audios[i].to(device=self.device, dtype=self.dtype),
                decoder_input_ids=decoder_input_ids.to(device=self.device),
                output_hidden_states=True,
                return_dict=True
            )
            
            encoder_last_hidden_state = model_output.encoder_last_hidden_state
            
            pooled_encoder_hidden_states = torch.nn.functional.adaptive_avg_pool1d(
                encoder_last_hidden_state.transpose(1, 2),  
                output_size=300  
            ).transpose(1, 2)  
            
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.squeeze(0)
            
            hidden_states.append(pooled_encoder_hidden_states)

        encoder_hidden_states = torch.stack(hidden_states, dim=0)  
        
        return encoder_hidden_states

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class WhisperAudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-small",
        num_projector_layers: int = 2,
        num_tokens_output: int = 300,
        emb_size: int = 768
    ):
        self.model_name_or_path = model_name_or_path
        self.module = WhisperAudioModule(model_name_or_path=self.model_name_or_path)
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output
        self.emb_size = emb_size
        
    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        return build_patch_mlp_projector(
            input_hidden_size=self.emb_size,
            lm_hidden_size=lm_hidden_size,
            num_layers=self.num_projector_layers,
        )    

    @property
    def name(self) -> str:
        return "audio_whisper"

    @property
    def token(self) -> str:
        return "<speech>"

    @property
    def data_key(self) -> str:
        return "speech_audios"

    @property
    def token_width(self) -> int:
        self.num_tokens_output = 300
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "WhisperAudioModality":
        self.module.to(dtype=dtype, device=device)
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
                audio_processed = self.module.feature_extractor(
                    audio_dict["array"],
                    return_tensors="pt",
                    sampling_rate=audio_dict["sampling_rate"],
                ).input_features 
                audios.append(audio_processed)
            row_values.append(torch.stack(audios) if len(audios) > 0 else None)
        return row_values 

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        audio_features = []
        
        for audio_batch in encoded_values:
            audio_features.append(self.module.forward(audio_batch))
        
        return audio_features 
