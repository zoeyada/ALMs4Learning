import json
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

test_data = "./data/L2-Arctic-plus/test_data.json"
with open(test_data, "r") as f:
    test_data = json.load(f)
    # test_data = test_data[:2]

# Load model and processor, qwen or qwen2

#  qwen2-audio
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True, device_map="auto")
output_path = "./pipelines/existing_alms/alms/results/qwen2.json"

# qwen-audio
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
# output_path = "./pipelines/existing_alms/results/qwen.json"

results = []

system_prompt = f'''
    You are a phonetics expert tasked with analyzing the pronunciation of audio and comparing it to the provided Ground Truth text.
    Your goal is to identify pronunciation issues, such as substitution, addition, or deletion of sounds, based on the audio input.

    Instructions:
    1. For each word in the Ground Truth, compare its pronunciation in the audio.
    2. Identify any mispronunciations and describe the issue (substitution, addition, deletion of sounds).
    3. For each issue, provide a suggestion using ARPAbet phonetic symbols.
    4. If the pronunciation is correct, simply output "No Problem".
    5. Do not include additional commentary. Just output the issues and suggestions for each word that has problems.

    Your task is to analyze the following audio and Ground Truth text for pronunciation issues and provide your suggestions.
'''
    
user_prompt = '''
    You are a phonetics expert. Your task is to detect mispronouciation based on given Ground Truth and Audio.
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
    Ground Truth: "{ground_truth}"''','''Output:'''

for entry in tqdm(test_data, desc="Processing audio files"):
    audio_path = entry["audio_path"]
    ground_truth = entry["text"]
    mis_exp_sug = entry["mis_exp_sug"]

    conversation = [
        {'role': 'system', 'content': system_prompt}, 
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt[0].format(ground_truth=ground_truth)},
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": user_prompt[1]}
        ]},
    ]
    
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if "audio_url" in ele:
                    audio_data, _ = librosa.load(
                        ele["audio_url"],
                        sr=processor.feature_extractor.sampling_rate
                    )
                    audios.append(audio_data)
    

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to("cuda")

    generated_ids = model.generate(**inputs, max_length=2048)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    results.append({
        "audio_path": audio_path,
        "text": ground_truth,
        "mis_exp_sug": mis_exp_sug,
        "speculative_mispronunciations": response
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4 ,ensure_ascii=False)

