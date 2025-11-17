import json
import torch
import string
import os

from pipelines.instruction_tuning.model.inference import load_trained_lora_model
from pipelines.instruction_tuning.model.data_tools import encode_chat

test_data = "./data/L2-Arctic-plus/test_data.json"

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
modality_builder = "audio_whisper"
modality_builder_type = "large"

model_name = "llama" if "llama" in model_name_or_path.lower() else "mistral"
asr_name = "wav2vec2" if "wav2vec2" in modality_builder else "whisper"

lora_path = f"./pipelines/instruction_tuning/checkpoints/finetuning/{model_name}_{asr_name}_{modality_builder_type}/checkpoint-105"
output_file = f"./pipelines/instruction_tuning/inference/results/{model_name}_{asr_name}_{modality_builder_type}.json"

model_cls_map = {
    "mistralai/Mistral-7B-Instruct-v0.1": "MistralLMMForCausalLM",
    "meta-llama/Llama-3.1-8B-Instruct": "LlamaLMMForCausalLM"
}
model_cls = model_cls_map.get(model_name_or_path, "LlamaLMMForCausalLM")

model, tokenizer = load_trained_lora_model(
    model_name_or_path=model_name_or_path,
    model_lora_path=lora_path,
    load_bits=16,
    modality_builder=modality_builder,
    modality_builder_type=modality_builder_type
)

PROMPT = '''Your task is to analyze the provided audio and compare it with the Ground Truth to identify pronunciation differences at the phoneme level.
The audio in <speech> contains a recording by a non-native English speaker. 

Below is the ground truth transcription:
Ground Truth: "{ground_truth}"

Output Format:
word: <mispronounced_word> issue: <issues> suggestion: <suggestions>
...

Output Rules:
1. Analyze each word in the Ground Truth and compare it with the pronunciation in the actual audio.
2. If the word in the Ground Truth has one or more pronunciation issues based on the audio:
    a. List the word from the Ground Truth.
    b. Combine all issues into a single line under "issue".
    c. Provide a single combined suggestion for correcting the issues using ARPAbet phonetic symbols.
3. If no errors are found in any of the Ground Truth words, output "No Problem". 
4. Do not output anything except for the words with pronunciation issues or "No Problem". 
5. Ensure the analysis focuses on the pronunciation of Ground Truth words as they appear in the audio.
6. Do not include any additional commentary outside of the analysis and suggestions.
7. Use ARPAbet symbols to describe phonetic issues.'''

def clean_text(text):
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace("'", "")
    return text.translate(str.maketrans("", "", punctuation_to_remove))

with open(test_data, "r") as f:
    data = json.load(f)
results = []

for i, entry in enumerate(data):
    audio_path = entry["audio_path"]
    text = clean_text(entry["text"])

    system_prompt = '''You are a phonetics expert tasked with identifying pronunciation differences between the provided Ground Truth and the corresponding pronunciation. 
    Analyze each word in the Ground Truth, identify pronunciation issues, and offer suggestions for improvement. '''        

    input_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PROMPT.format(ground_truth=text)},
        ],
        "speech_audios": [audio_path],
    }

    encoded_dict = encode_chat(input_data, tokenizer, model.modalities, model_cls)
    input_ids = torch.tensor(encoded_dict["input_ids"]).unsqueeze(0).to(model.device)
    modality_inputs = {m.name: [encoded_dict[m.name]] for m in model.modalities}

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.01,
            modality_inputs=modality_inputs,
        )

    output = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    entry["speculative_mispronunciations"] = output
    results.append(entry)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

