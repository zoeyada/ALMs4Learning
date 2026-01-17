import base64
import json
from openai import OpenAI
from tqdm import tqdm
import os


test_data = "./data/L2-Arctic-plus/test_data.json"
with open(test_data, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    # test_data = test_data[:2]

api_key = "<API_KEY>" # replace with your API key
client = OpenAI(api_key=api_key)
    
output_path = "./pipelines/existing_alms/alms/results/gpt4o.json"
results = []

tqdm_iter = tqdm(test_data, desc="Processing", unit="file")

for entry in tqdm_iter:
    audio_path = entry['audio_path']
    ground_truth = entry['text']

    with open("./data/L2-Arctic-plus/l2arctic_release_v5.0/ABA/wav/arctic_a0501.wav", "rb") as audio_file:
        wav_data = audio_file.read()
        encoded_string_example = base64.b64encode(wav_data).decode('utf-8')

    with open(audio_path, "rb") as audio_file:
        wav_data = audio_file.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

    system_prompt = f'''
You are a phonetics expert tasked with identifying pronunciation differences between the provided Ground Truth and the corresponding pronunciation. 
Analyze each word in the Ground Truth, identify pronunciation issues, and offer suggestions for improvement. 
    '''
        
    user_prompt = '''
You are a phonetics expert. Your task is to detect mispronouciation based on given Ground Truth and Audio.
This is an example of the format you should use and some output rules you should follow.
        
Output Format:
word: <one_word_in_ground_truth> issue: <issues> suggestion: <suggestions>
word: <one_word_in_ground_truth> issue: <issues> suggestion: <suggestions>
…
        
Output Rules:
1. Analyze each word in the Ground Truth and compare it with the pronunciation in the actual audio.
2. If the word in the Ground Truth has one or more pronunciation issues based on the audio:
    a. List the word from the Ground Truth.
    b. Combine all issues into a single line under "issue".
    c. Provide a single combined suggestion for correcting the issues using ARPAbet phonetic symbols.
3. If no errors are found in any of the Ground Truth words, output "No Problem". But there is a high probability of pronunciation problems.
4. Do not output anything except for the words with pronunciation issues or "No Problem". 
5. Ensure the analysis focuses on the pronunciation of Ground Truth words as they appear in the audio.
6. Do not include any additional commentary outside of the analysis and suggestions.
7. Use ARPAbet symbols to describe phonetic issues.

Here is an example of how you should analyze pronunciation based on the audio and the Ground Truth text. 
        
Input:
Ground Truth: "you're joking me sir the other managed to articulate"
Audio: ''','''
Output:
word: joking issue: "JH" was replaced with "ZH", indicating a substitution error. An extra "G" sound was added, indicating an addition error. An extra "AH" sound was added, indicating an addition error. suggestion: Practice the difference between /JH/ as in "JOKE" (/JH OW K/) and /ZH/ as in "MEASURE" (/M EH ZH ER/). Focus on stopping after the /NG/ as in "KING" (/K IH NG/) without additional sounds. Avoid adding extra vowel sounds after completing the word.
word: sir issue: Unclear pronunciation, "ER" perceived with uncertainty suggestion: Practice /ER/ as in "SIR" (/S ER/) to add clarity
word: other issue: "DH" was replaced with "Z", indicating a substitution error. Unclear pronunciation, "ER" perceived with uncertainty. suggestion: Practice unvoiced /DH/ as in "THIS" (/DH IH S/) instead of voiced consonant sounds like /Z/. Practice /ER/ as in "HER" (/HH ER/) for more distinct articulation.
word: managed issue: "JH" was replaced with "ZH", indicating a substitution error suggestion: Practice the distinction between /JH/ as in "JUDGE" (/JH AH JH/) and /ZH/ as in "VISION" (/V IH ZH UH N/)
word: articulate issue: "R" was replaced with a foreign-accented "R*", indicating a substitution error. "EY" was replaced with "EH", indicating a substitution error. suggestion: Practice the American /R/ sound as in "RED" (/R EH D/) emphasizing the retroflex position of the tongue. Practice the distinction between /EY/ as in "DATE" (/D EY T/) and /EH/ as in "BET" (/B EH T/)
        
Input:
Ground Truth: {ground_truth}
Audio: ''','''
Output:
'''

    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
    messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text",
                        "text": user_prompt[0]
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string_example,
                            "format": "wav"
                        }
                    },
                    { 
                        "type": "text",
                        "text": user_prompt[1].format(ground_truth=ground_truth)
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav"
                        }
                    },
                    { 
                        "type": "text",
                        "text": user_prompt[2]
                    },
                ]
            },
        ]
    )

    content = completion.choices[0].message.content

    results.append({
        "audio_path": audio_path,
        "text": ground_truth,
        "mis_exp_sug": entry.get("mis_exp_sug", {}),
        "speculative_mispronunciations": content
    })

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)