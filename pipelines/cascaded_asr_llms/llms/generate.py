from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import os
import re

input_files = [
    "./pipelines/cascaded_asr_llms/asr/results/test/wav2vec2-base-960h.json",
    "./pipelines/cascaded_asr_llms/asr/results/test/wav2vec2-large-960h-lv60-self.json",
    "./pipelines/cascaded_asr_llms/asr/results/test/whisper-small.json",
    "./pipelines/cascaded_asr_llms/asr/results/test/whisper-medium.json",
    "./pipelines/cascaded_asr_llms/asr/results/test/whisper-large.json"
]

output_base_path = "./pipelines/cascaded_asr_llms/llms/results/"

model_dict = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

system_prompt = f'''
    You are a phonetics expert tasked with identifying pronunciation differences between the provided Ground Truth and the corresponding pronunciation. 
    Analyze each word in the Ground Truth, identify pronunciation issues, and offer suggestions for improvement. 
    '''
    
user_prompt = '''
    You are a phonetics expert. Your task is to compare the provided Transcribed Text with the Ground Truth transcription. Identify any pronunciation differences for each word in the Ground Truth based on the transcription and provide specific suggestions for improvement. 
    
    Input:
    Ground Truth: "{ground_truth}"
    Transcribed Text: "{transcribed_text}"
    
    Output Format: 
    word: <word_in_ground_truth>
    issue: <issues>
    suggestion: <suggestions>
    ...

    Output Rules:
    1. Analyze each word in the Ground Truth and compare it with the corresponding word in the Transcribed Text.
    2. For each word in the Ground Truth, output:
        word: <word_in_ground_truth>
        issue: <issues> (if there are pronunciation issues)
        suggestion: <suggestions> (if there are pronunciation issues)
       If there are no issues with a word, output:
        word: <word_in_ground_truth>
        issue: None
        suggestion: None
    3. If a word has multiple issues, combine them into a single issue line and provide a single combined suggestion for correction.
    4. Do not include any additional commentary outside of the analysis and suggestions.
    5. Use ARPAbet phonetic symbols to describe the pronunciation issues.

    Example Input:
    Ground Truth: you're joking me sir the other managed to articulate
    Transcribed Text: your soking me ser the other managed to articulate

    Example Output:
    word: you're
    issue: None
    suggestion: None
    word: joking
    issue: "JH" was replaced with "ZH", indicating a substitution error. An extra "G" sound was added, indicating an addition error. An extra "AH" sound was added, indicating an addition error.
    suggestion: Practice the difference between /JH/ as in "JOKE" (/JH OW K/) and /ZH/ as in "MEASURE" (/M EH ZH ER/). Focus on stopping after the /NG/ as in "KING" (/K IH NG/) without additional sounds. Avoid adding extra vowel sounds after completing the word.
    word: me
    issue: None
    suggestion: None
    word: sir
    issue: Unclear pronunciation, "ER" perceived with uncertainty
    suggestion: Practice /ER/ as in "SIR" (/S ER/) to add clarity
    word: the
    issue: None
    suggestion: None
    word: other
    issue: "DH" was replaced with "Z", indicating a substitution error. Unclear pronunciation, "ER" perceived with uncertainty.
    suggestion: Practice unvoiced /DH/ as in "THIS" (/DH IH S/) instead of voiced consonant sounds like /Z/. Practice /ER/ as in "HER" (/HH ER/) for more distinct articulation.
    word: managed
    issue: "JH" was replaced with "ZH", indicating a substitution error
    suggestion: Practice the distinction between /JH/ as in "JUDGE" (/JH AH JH/) and /ZH/ as in "VISION" (/V IH ZH UH N/)
    word: to
    issue: None
    suggestion: None
    word: articulate
    issue: "R" was replaced with a foreign-accented "R*", indicating a substitution error. "EY" was replaced with "EH", indicating a substitution error.
    suggestion: Practice the American /R/ sound as in "RED" (/R EH D/) emphasizing the retroflex position of the tongue. Practice the distinction between /EY/ as in "DATE" (/D EY T/) and /EH/ as in "BET" (/B EH T/)
'''

def generate_error_detection(entry, model, tokenizer):
    audio_path = entry["audio_path"]
    text = entry["text"]
    transcribed_text = entry["transcribed_text"]
    mis_exp_sug = entry["mis_exp_sug"]

    message_content = user_prompt.format(
        ground_truth = text, 
        transcribed_text = transcribed_text
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": message_content
        }
    ]

    error_detection = None

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    error_detection = tokenizer.decode(response, skip_special_tokens=True)
        
    return {
        "audio_path": audio_path,
        "text": text,
        "transcribed_text": transcribed_text,
        "mis_exp_sug": mis_exp_sug,
        "speculative_mispronunciations": error_detection,
    }

def process_data(input_file, output_file, model, tokenizer):
    with open(input_file, "r") as f:
        transcription_data = json.load(f)

    error_detection_results = []

    for entry in transcription_data:
        text = entry['text']
        text = re.sub(r'[,!?.:"]', '', text)
        text = re.sub(r'-', ' ', text)
        text = text.lower().strip()
        entry['text'] = text

        transcribed_text = entry['transcribed_text']
        number_to_word = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }

        for number, word in number_to_word.items():
            transcribed_text = transcribed_text.replace(number, word + ' ')
        transcribed_text = transcribed_text.lower()
        transcribed_text = re.sub(r'[,.!?":;-]', ' ', transcribed_text)
        transcribed_text = re.sub(r'\s+', ' ', transcribed_text).strip()

        entry['transcribed_text'] = transcribed_text

    for entry in tqdm(transcription_data, desc=f"{output_file}"):
        result = generate_error_detection(entry, model, tokenizer)
        error_detection_results.append(result)

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(error_detection_results, f, indent=4, ensure_ascii=False)



for model_name, model_id in model_dict.items():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    for input_file in input_files:
        input_filename = os.path.basename(input_file)
        output_file = os.path.join(output_base_path, model_name, input_filename)

        process_data(input_file, output_file, model, tokenizer)

    del model
    torch.cuda.empty_cache()