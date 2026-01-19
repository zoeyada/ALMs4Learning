import json
import re
import os

# Put your files that need to be parsed here (only for instruction tuning pipeline)
# The default input base path is "./pipelines/instruction_tuning/inference/results" 
# FILES_TO_PROCESS = [
#     "/data2/cuizhouying/ALMs4Learning/pipelines/instruction_tuning/inference/results/mistral_whisper_small.json",
# ]

# OUTPUT_BASE = "/home/cuizhouying/ALMs4Learning/pipelines/instruction_tuning/parse/results"

FILES_TO_PROCESS = [
    "./pipelines/instruction_tuning/inference/results/mistral_whisper_small.json",
]

OUTPUT_BASE = "./pipelines/instruction_tuning/parse/results"
def parse_speculative_mispronunciations(entry):
    mis_exp_sug = {}
    raw_text = entry.get("speculative_mispronunciations", "")

    if raw_text == "No Problem":
        return mis_exp_sug

    matches = re.findall(
        r"word:\s*(.*?)\s*issue:\s*(.*?)\s*suggestion:\s*(.*?)\s*(?=\s*word:|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE
    )

    for word, issue, suggestion in matches:
        word = word.strip()
        issue = issue.strip()
        suggestion = suggestion.strip()

        if " " in word:
            continue

        if word not in mis_exp_sug:
            mis_exp_sug[word] = []

        if not any(item["issue"] == issue and item["suggestion"] == suggestion for item in mis_exp_sug[word]):
            mis_exp_sug[word].append({
                "issue": issue,
                "suggestion": suggestion
            })

    return {k: v for k, v in mis_exp_sug.items() if v}

def process_file(input_file, output_file):
    with open(input_file, "r") as f:
        input_data = json.load(f)

    output_data = []
    for entry in input_data:
        parsed_entry = {
            "audio_path": entry.get("audio_path"),
            "text": entry.get("text"),
            "mis_exp_sug": entry.get("mis_exp_sug"),
            "speculative_mispronunciations": parse_speculative_mispronunciations(entry)
        }
        output_data.append(parsed_entry)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def main():
    for input_file in FILES_TO_PROCESS:
        file_name = os.path.basename(input_file)
        output_file = os.path.join(OUTPUT_BASE, file_name)
        process_file(input_file, output_file)


if __name__ == "__main__":
    main()
