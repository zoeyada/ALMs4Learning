from typing import List, Dict
from abc import ABC, abstractmethod

from torch.nn.functional import conv1d
import torch
import logging

from pipelines.instruction_tuning.model.modalities.base_modality import Modality

class LMMMetaModel:
    def __init__(self, config):
        super(LMMMetaModel, self).__init__(config)

    def _load_projector_weights(self, weights: Dict):
        weights = {
            (k[23:] if k.startswith("base_model.model.model.") else k): v
            for k, v in weights.items()
        }

        logging.info(f"Loading pretrained weights: {list(weights.keys())}")
        keyword = self.config.modality_builder + "_lmm_projector"
               
        weights = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        # print("weights keys:", weights.keys()," keyword:", keyword)
        # exit()
        if (self.config.modality_builder == "audio_whisper"):
            self.audio_whisper_lmm_projector.load_state_dict(weights, strict=False)
        if (self.config.modality_builder == "audio_wav2vec2"):
            self.audio_wav2vec2_lmm_projector.load_state_dict(weights, strict=False)


    def initialize_pretrained_modules(self, modalities: List[Modality], weights: Dict):
        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)
        # print("modalities:", [m.name for m in modalities])
        # exit()
        self._load_projector_weights(weights)

    def initialize_modules(self, modalities: List[Modality], weights: Dict):
        names = [m.name for m in modalities]

        self.config.modalities = names

        for m in modalities:
            projector = m.build_projector(self.config.hidden_size)
            setattr(self, m.name + "_lmm_projector", projector)
        # print("modalities:", [m.name for m in modalities])
        # exit()   
        self._load_projector_weights(weights)


class LMMMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> "LMMMetaForCausalLM":
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        model = self.get_model()

        batch_size, seq_len = input_ids.shape

        # batch_size x seq_len x embedding_hidden_size
        # (1, 50, 4096)
        inputs_embeds = torch.zeros(
            (batch_size, seq_len, self.config.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )

        # modality x batch_size x instance_idx x modality_token_width x embedding_hidden_size
        projected_tensors = []
        # assuming that if caching is enabled, we'll never have past_key_values AND need to encode the instruction modality values

        # if past_key_values is None or len(past_key_values.key_cache)==0 or len(projected_tensors)==0:
        if past_key_values is None or len(past_key_values.key_cache)==0:
            # print("self.modalities:", self.modalities)
            # device = self.device
            for m in self.modalities:
                # print("modalities_name:", m.name) # audio_whisper
                m_vals = m.forward(kwargs.get(m.name))
                mp_vals = []
                # print("m.name:", m.name, "model:", model)
                
                proj = getattr(model, m.name + "_lmm_projector")
                

                # project each batch into language model token space
                for m_val in m_vals:
                    # m_val = m_val.to(device)
                    mp_vals.append(proj(m_val))
                    
                #for mp_val in mp_vals:    
                    # print("mp_val.shape[1:]:", mp_val.shape[1:], "(m.token_width, self.config.hidden_size),:", (m.token_width, self.config.hidden_size))
                # mp_val.shape[1:]: torch.Size([1, 13, 4096]) (m.token_width, self.config.hidden_size),: (10, 4096)
                # 8 times # per_device_train_batch_size
                
                # assert all(
                #     mp_val.shape[1:] == (m.token_width, self.config.hidden_size)
                #     for mp_val in mp_vals
                # ), (
                #     "Modality tensors have incorrect shape, check your projector implementation "
                #     + str([mp_val.shape[1:] for mp_val in mp_vals])
                #     + " vs expected "
                #     + str((m.token_width, self.config.hidden_size)) # 10， 4096
                # )
                
                projected_tensors.append(mp_vals)

        indices = None
        for i, input_ids_sample in enumerate(input_ids):
            is_text_mask = input_ids_sample >= 0

            # fill in all the LLM-based text embeddings
            inputs_embeds[i, is_text_mask] = model.embed_tokens(
                input_ids_sample[is_text_mask]
            )

            # print(past_key_values)
            # print(len(past_key_values.key_cache))
            
            # skip if all tokens are text tokens
            if is_text_mask.sum() == seq_len:
                continue
            assert(
                past_key_values is None or len(past_key_values.key_cache)==0 # past_key_values.key_cache
                # past_key_values is None
            ), "We shouldn't have cached keys if this is the first instruction pass"

            for mi, m in enumerate(self.modalities):
                # locate the group of tokens for this modality
                m_mask = (input_ids_sample == m.token_idx).float() # -8565
                m_kernel = torch.tensor(
                    [-1] * m.token_width, dtype=m_mask.dtype, device=m_mask.device
                ) # tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.], device='cuda:0')
                
                m_conv = conv1d(
                    m_mask.unsqueeze(0).unsqueeze(0),
                    m_kernel.unsqueeze(0).unsqueeze(0),
                )
                
                indices = (m_conv[0, 0] == -m.token_width).nonzero(as_tuple=True)[0]
                # print("indices:", indices)
                
                # fill these embeddings with the projected modality tensor
                last_covered_idx = -1 # 12, beginning of the speech token
                k = 0
                for possible_token_idx in indices:
                    if possible_token_idx <= last_covered_idx:
                        # make sure we don't overwrite an instance we've already covered
                        # handles bug caused by back-to-back tokens
                        continue
                    
                    batch_modality_tensor = projected_tensors[mi][i][k]
                    inputs_embeds[
                        # i, possible_token_idx : possible_token_idx + m.token_width
                        i, possible_token_idx : possible_token_idx + m.token_width
                    ] = batch_modality_tensor
                    last_covered_idx = possible_token_idx + m.token_width - 1
                    k += 1
                    
                    # # projected_tensors[mi][0]
                    # batch_modality_tensor = projected_tensors[mi][0]  
                    # inputs_embeds[i, possible_token_idx : possible_token_idx + m.token_width] = batch_modality_tensor
                    # last_covered_idx = possible_token_idx + m.token_width - 1

        return None, attention_mask, past_key_values, inputs_embeds, labels
