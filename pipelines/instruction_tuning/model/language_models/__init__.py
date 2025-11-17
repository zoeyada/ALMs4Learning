from pipelines.instruction_tuning.model.language_models.mistral import (
    MistralLMMForCausalLM,
)

from pipelines.instruction_tuning.model.language_models.llama import LlamaLMMForCausalLM

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM, LlamaLMMForCausalLM]

LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}
