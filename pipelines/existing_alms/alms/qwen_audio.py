import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm  

# qwen-audio
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
test_data = "./data/L2-Arctic-plus/test_data.json"
output_path = "./pipelines/existing_alms/alms/results/qwen.json"

output_results = []

with open(test_data, "r") as f:
    data = json.load(f)
    # data = data[:2]  


for entry in tqdm(data, desc="Processing entries"):
    audio_path = entry.get("audio_path")
    text = entry.get("text")
    mis_exp_sug = entry.get("mis_exp_sug")
    
    prompt = '''You are a phonetics expert. Your goal is to identify pronunciation issues, such as substitution, addition, or deletion of sounds, based on the audio input and Audio.
    This is an example of the format you should use and some output rules you should follow. 
        
    Output Format:
    word: <word_in_ground_truth> issue: <issues> suggestion: <suggestions>
    word: <word_in_ground_truth> issue: <issues> suggestion: <suggestions>
    …
        
    Output Rules:
    1. Analyze each word in the Ground Truth and compare it with the pronunciation in the actual audio.
    2. If the word in the Ground Truth has one or more pronunciation issues based on the audio:
        a. List the word from the Ground Truth.
        b. Combine all issues into a single line under "issue".
        c. Provide a single combined suggestion for correcting the issues using ARPAbet phonetic symbols.
    3. Ensure the analysis focuses on the pronunciation of Ground Truth words as they appear in the audio.
    4. Do not include any additional commentary outside of the analysis and suggestions. Just begin with the first mispronunced word, instead of using 'Output:'.
    5. Use ARPAbet symbols and English to describe phonetic issues. 
    6. If there are no issues with the words in the Ground Truth, output 'No Problem'. "No Problem" should appear on its own and cannot be included as part of the issue or suggestion.

    Here is an example of how you should analyze pronunciation based on the audio and the Ground Truth text. 
        
    Input:
    Ground Truth: "you're joking me sir the other managed to articulate"

    Output:
    word: joking issue: "JH" was replaced with "ZH", indicating a substitution error. An extra "G" sound was added, indicating an addition error. An extra "AH" sound was added, indicating an addition error. suggestion: Practice the difference between /JH/ as in "JOKE" (/JH OW K/) and /ZH/ as in "MEASURE" (/M EH ZH ER/). Focus on stopping after the /NG/ as in "KING" (/K IH NG/) without additional sounds. Avoid adding extra vowel sounds after completing the word.
    word: sir issue: Unclear pronunciation, "ER" perceived with uncertainty suggestion: Practice /ER/ as in "SIR" (/S ER/) to add clarity
    word: other issue: "DH" was replaced with "Z", indicating a substitution error. Unclear pronunciation, "ER" perceived with uncertainty. suggestion: Practice unvoiced /DH/ as in "THIS" (/DH IH S/) instead of voiced consonant sounds like /Z/. Practice /ER/ as in "HER" (/HH ER/) for more distinct articulation.
    word: managed issue: "JH" was replaced with "ZH", indicating a substitution error suggestion: Practice the distinction between /JH/ as in "JUDGE" (/JH AH JH/) and /ZH/ as in "VISION" (/V IH ZH UH N/)
    word: articulate issue: "R" was replaced with a foreign-accented "R*", indicating a substitution error. "EY" was replaced with "EH", indicating a substitution error. suggestion: Practice the American /R/ sound as in "RED" (/R EH D/) emphasizing the retroflex position of the tongue. Practice the distinction between /EY/ as in "DATE" (/D EY T/) and /EH/ as in "BET" (/B EH T/)
        
    Input:
    Ground Truth: "{ground_truth}"
    Output:'''
    
    query = tokenizer.from_list_format([
        {'audio': audio_path},
        {'text': prompt.format(ground_truth=text)},
    ])

    try:
        response, history = model.chat(tokenizer, query=query, history=None)
        output_results.append({
            "audio_path": audio_path,
            "text": text,
            "mis_exp_sug": mis_exp_sug,
            "speculative_mispronunciations": response
        })
    except Exception as e:
        print(f"Error processing entry with audio_path {audio_path}: {e}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output_results, f, indent=4, ensure_ascii=False)