from typing import Type, List, Optional
import logging

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from peft import PeftModel
import torch
import os

from pipelines.instruction_tuning.model.model_utils import fix_tokenizer
from pipelines.instruction_tuning.model.modalities.base_modality import Modality
from pipelines.instruction_tuning.model.language_models.mistral import MistralForCausalLM
from pipelines.instruction_tuning.model.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from pipelines.instruction_tuning.model.modalities import MODALITY_BUILDERS
from pipelines.instruction_tuning.model.modalities.audio_wav2vec2 import Wav2Vec2AudioModality
from pipelines.instruction_tuning.model.modalities.audio_whisper import WhisperAudioModality

def load_trained_lora_model(
    model_name_or_path: str,
    model_lora_path: str,
    model_cls: Optional[Type] = None,
    modalities: Optional[List[Modality]] = None,
    load_bits: int = 16,
    device_map: str = "auto",
    modality_builder = "audio_whisper",
    modality_builder_type = "small"

):
    load_kwargs = {"device_map": device_map}

    if load_bits == 8:
        load_kwargs["load_in_8bit"] = True
    elif load_bits == 4:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_bits == 16:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Invalid load_bits: {load_bits}")
    
    load_kwargs["attn_implementation"] = "flash_attention_2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    fix_tokenizer(tokenizer)

    cfg = AutoConfig.from_pretrained(model_lora_path)
    if model_cls is None:
        model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[cfg.model_cls]
    # if modalities is None:
    #     modalities = MODALITY_BUILDERS[cfg.modality_builder]()
    
    if modalities is None:
        modalities = []
        if (modality_builder == "audio_whisper"):
            if (modality_builder_type == "small"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-small", 
                        emb_size=768
                    )]
            elif (modality_builder_type == "medium"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-medium", 
                        emb_size=1024
                    )]            
            elif (modality_builder_type == "large"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-large", 
                        emb_size=1280
                    )]
        
        elif (modality_builder == "audio_wav2vec2"):
            if (modality_builder_type == "base"):
                modalities = [Wav2Vec2AudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="facebook/wav2vec2-base-960h",
                    emb_size=768
                )]   
            elif (modality_builder_type == "large"):
                modalities = [Wav2Vec2AudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="facebook/wav2vec2-large-960h-lv60-self",
                    emb_size=1024
                )]          
        else:
            modalities = MODALITY_BUILDERS[modality_builder]()

    logging.info(f"Loading base model from {model_name_or_path} as {load_bits} bits")
    
    print(model_name_or_path)
    model = model_cls.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, config=cfg, **load_kwargs
    )
    model.modalities = modalities

    logging.info(f"Loading projector weights for {[m.name for m in modalities]}")
    if os.path.exists(os.path.join(model_lora_path, "non_lora_trainables.bin")):
        non_lora_trainables = torch.load(
            os.path.join(model_lora_path, "non_lora_trainables.bin"), map_location="cpu"
        )
    else:
        local_fn = hf_hub_download(
            repo_id=model_lora_path,
            filename="non_lora_trainables.bin",
            repo_type="model",
        )
        non_lora_trainables = torch.load(local_fn, map_location="cpu")
    model.get_model().initialize_pretrained_modules(modalities, non_lora_trainables)

    logging.info(f"Loading and merging LoRA weights from {model_lora_path}")
    model = PeftModel.from_pretrained(model, model_lora_path)
    
    if load_bits == 16:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.data = param.data.to(torch.bfloat16)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(torch.bfloat16)    

    if load_bits == 16:
        # TODO: Figure out why this fails for other bit sizes
        model = model.merge_and_unload()
    model.eval()

    return model, tokenizer

def load_pretrained_model(
    model_name_or_path: str,
    model_pretrain_path: str,
    model_cls: Optional[Type] = None,
    modalities: Optional[List[Modality]] = None,
    load_bits: int = 16,
    device_map: str = "auto",
    modality_builder = "audio_whisper",
    modality_builder_type = "small"
):
    load_kwargs = {"device_map": device_map}

    if load_bits == 8:
        load_kwargs["load_in_8bit"] = True
    elif load_bits == 4:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_bits == 16:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Invalid load_bits: {load_bits}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    fix_tokenizer(tokenizer)

    cfg = AutoConfig.from_pretrained(model_pretrain_path)
    if model_cls is None:
        model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[cfg.model_cls]

    if modalities is None:
        modalities = []
        if (modality_builder == "audio_whisper"):
            if (modality_builder_type == "small"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-small", 
                        emb_size=768
                    )]
            elif (modality_builder_type == "medium"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-medium", 
                        emb_size=1024
                    )]            
            elif (modality_builder_type == "large"):
                modalities = [WhisperAudioModality(
                        num_tokens_output=300, 
                        model_name_or_path="openai/whisper-large", 
                        emb_size=1280
                    )]
        
        elif (modality_builder == "audio_wav2vec2"):
            if (modality_builder_type == "base"):
                modalities = [Wav2Vec2AudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="facebook/wav2vec2-base-960h",
                    emb_size=768
                )]   
            elif (modality_builder_type == "large"):
                modalities = [Wav2Vec2AudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="facebook/wav2vec2-large-960h-lv60-self",
                    emb_size=1024
                )]          
        else:
            modalities = MODALITY_BUILDERS[modality_builder]()   

    logging.info(f"Loading base model from {model_name_or_path} as {load_bits} bits")
    model = model_cls.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, config=cfg, **load_kwargs, 
    )
    model.modalities = modalities

    logging.info(f"Loading projector weights for {[m.name for m in modalities]}")
    if os.path.exists(os.path.join(model_pretrain_path, "non_lora_trainables.bin")):
        non_lora_trainables = torch.load(
            os.path.join(model_pretrain_path, "non_lora_trainables.bin"), map_location="cpu"
        )
    else:
        local_fn = hf_hub_download(
            repo_id=model_pretrain_path,
            filename="non_lora_trainables.bin",
            repo_type="model",
        )
        non_lora_trainables = torch.load(local_fn, map_location="cpu")
    model.get_model().initialize_pretrained_modules(modalities, non_lora_trainables)
    # model.projector = load_results
    model.eval()

    return model, tokenizer
