from typing import List
import argparse
import json
import os
from datasets import Dataset

# Reminder: Please specify the correct output folder path.
# It is recommended to save your output in a subfolder within the default directory "./data/training_datasets/finetune".

test_data = "./data/L2-Arctic-plus/train_data.json"

def _load_data_from_json():
    with open(test_data, "r") as f:
        data = json.load(f)
    return data

system_prompt = '''You are a phonetics expert tasked with identifying pronunciation differences between the provided Ground Truth and the corresponding pronunciation. 
Analyze each word in the Ground Truth, identify pronunciation issues, and offer suggestions for improvement. '''

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
7. Use ARPAbet symbols to describe phonetic issues.
'''

def format_issues(mis_exp_sug: dict) -> str:
    if not mis_exp_sug:
        return "No Problem"

    formatted = []
    for word, issue_data in mis_exp_sug.items():
        issues, suggestions = [], []
        for issue in issue_data:
            issue_description = issue.get("issue", "")
            suggestion = issue.get("suggestion", "")
            if issue_description and not issue_description.endswith('.'):
                issue_description += '.'
            if suggestion and not suggestion.endswith('.'):
                suggestion += '.'
            if issue_description:
                issues.append(issue_description)
            if suggestion:
                suggestions.append(suggestion)
        formatted.append(f"word: {word} issue: {' '.join(issues)} suggestion: {' '.join(suggestions)}")

    return "\n".join(formatted)

def _build_convo(row) -> dict:
    ground_truth = row.get("text", "")
    if not ground_truth:
        raise ValueError("Ground truth text is missing in row")

    q = PROMPT.format(ground_truth=ground_truth)
    mis_exp_sug = row.get("mis_exp_sug", {})
    formatted_issues = format_issues(mis_exp_sug)

    return {
        "speech_audios": [{"audio_path": row["audio_path"]}],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": formatted_issues},
        ],
    }

def main(args):
    data = _load_data_from_json()

    def gen():
        for row in data:
            try:
                yield _build_convo(row)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

    ds = Dataset.from_generator(gen, num_proc=args.num_proc)
    ds.save_to_disk(args.output_folder)

    print(ds)
    print(len(ds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="./data/training_datasets/finetune",
    )
    parser.add_argument("-p", "--num_proc", type=int, default=10)
    args = parser.parse_args()
    main(args)