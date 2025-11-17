from pipelines.instruction_tuning.model.modalities.audio_whisper import WhisperAudioModality
from pipelines.instruction_tuning.model.modalities.audio_wav2vec2 import Wav2Vec2AudioModality

MODALITY_BUILDERS = {
    "audio_whisper": lambda: [
        WhisperAudioModality(
            num_tokens_output=300, model_name_or_path="openai/whisper-large", emb_size=1280
        )
    ],
    "audio_wav2vec2":lambda: [
        Wav2Vec2AudioModality(
            num_tokens_output=300, model_name_or_path="facebook/wav2vec2-base-960h",emb_size = 768
        )
    ],
}
