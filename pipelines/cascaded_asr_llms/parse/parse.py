import json
import re
import os

# Put your files that need to be parsed here (only for cascaded ASR + LLMs pipeline)
# The default input base path is "./pipelines/cascaded_asr_llms/llms/results/{llm_name}/{asr_model_name}.json" 
FILES_TO_PROCESS = [
    # llama
    "./pipelines/cascaded_asr_llms/llms/results/llama/wav2vec2-base-960h.json",
    "./pipelines/cascaded_asr_llms/llms/results/llama/wav2vec2-large-960h-lv60-self.json",
    "./pipelines/cascaded_asr_llms/llms/results/llama/whisper-large.json",
    "./pipelines/cascaded_asr_llms/llms/results/llama/whisper-medium.json",
    "./pipelines/cascaded_asr_llms/llms/results/llama/whisper-small.json",
    
    # mistral
    "./pipelines/cascaded_asr_llms/llms/results/mistral/wav2vec2-base-960h.json",
    "./pipelines/cascaded_asr_llms/llms/results/mistral/wav2vec2-large-960h-lv60-self.json",
    "./pipelines/cascaded_asr_llms/llms/results/mistral/whisper-large.json",
    "./pipelines/cascaded_asr_llms/llms/results/mistral/whisper-medium.json",
    "./pipelines/cascaded_asr_llms/llms/results/mistral/whisper-small.json",
]

# The default output base path is "./pipelines/cascaded_asr_llms/parse/results/{llm_name}/"
OUTPUT_BASE = "./pipelines/cascaded_asr_llms/parse/results"

def parse_speculative_mispronunciations(entry):
    mis_exp_sug = {}
    raw_text = entry.get("speculative_mispronunciations", "")

    matches = re.findall(
        r"word:\s*(.*?)\n\s*issue:\s*(.*?)\n\s*suggestion:\s*(.*?)\n",
        raw_text,
        re.DOTALL | re.IGNORECASE
    )

    for word, issue, suggestion in matches:
        word, issue, suggestion = word.strip(), issue.strip(), suggestion.strip()

        if not issue or issue.lower() == "none" or not suggestion or suggestion.lower() == "none":
            continue
        if "None" in issue or "None" in suggestion:
            continue

        if word not in mis_exp_sug:
            mis_exp_sug[word] = []

        if not any(item['issue'] == issue and item['suggestion'] == suggestion for item in mis_exp_sug[word]):
            mis_exp_sug[word].append({
                "issue": issue,
                "suggestion": suggestion
            })

    return {k: v for k, v in mis_exp_sug.items() if v}

def process_file(input_file):
    llm_name = os.path.basename(os.path.dirname(input_file))  
    output_dir = os.path.join(OUTPUT_BASE, llm_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as f:
        input_data = json.load(f)

    output_data = []
    for entry in input_data:
        parsed_entry = {
            "audio_path": entry.get("audio_path"),
            "text": entry.get("text"),
            "transcribed_text": entry.get("transcribed_text"),
            "mis_exp_sug": entry.get("mis_exp_sug"),
            "speculative_mispronunciations": parse_speculative_mispronunciations(entry)
        }
        output_data.append(parsed_entry)

    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def main():
    for file_path in FILES_TO_PROCESS:
        process_file(file_path)

if __name__ == "__main__":
    main()
