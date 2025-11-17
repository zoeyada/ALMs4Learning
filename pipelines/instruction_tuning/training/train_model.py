import transformers
import logging
from pipelines.instruction_tuning.model.modalities.audio_whisper import WhisperAudioModality
from pipelines.instruction_tuning.model.modalities.audio_wav2vec2 import Wav2Vec2AudioModality

from pipelines.instruction_tuning.model.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from pipelines.instruction_tuning.model.training_data import (
    DataArguments,
)
from pipelines.instruction_tuning.model.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from pipelines.instruction_tuning.model.modalities import MODALITY_BUILDERS
import os

os.environ['WANDB_MODE'] = 'disabled'

from pytorch_lightning import seed_everything # Add seed_everything 


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    seed_everything(42)
    
    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    modalities = []
    
    if (model_args.modality_builder == "audio_whisper"):
        if (model_args.modality_builder_type == "small"):
            modalities = [WhisperAudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="openai/whisper-small", 
                    emb_size=768
                )]
        elif (model_args.modality_builder_type == "medium"):
            modalities = [WhisperAudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="openai/whisper-medium", 
                    emb_size=1024
                )]            
        elif (model_args.modality_builder_type == "large"):
            modalities = [WhisperAudioModality(
                    num_tokens_output=300, 
                    model_name_or_path="openai/whisper-large", 
                    emb_size=1280
                )]
    
    elif (model_args.modality_builder == "audio_wav2vec2"):
        if (model_args.modality_builder_type == "base"):
            modalities = [Wav2Vec2AudioModality(
                num_tokens_output=300, 
                model_name_or_path="facebook/wav2vec2-base-960h",
                emb_size=768
            )]   
        elif (model_args.modality_builder_type == "large"):
            modalities = [Wav2Vec2AudioModality(
                num_tokens_output=300, 
                model_name_or_path="facebook/wav2vec2-large-960h-lv60-self",
                emb_size=1024  
            )]          
    else:
        modalities = MODALITY_BUILDERS[model_args.modality_builder]()        

    # modalities = MODALITY_BUILDERS[model_args.modality_builder]()
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_modalities(model_cls, training_args, model_args, data_args, modalities)