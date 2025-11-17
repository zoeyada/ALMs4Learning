import json
from tqdm import tqdm
import torch
import jiwer
import Levenshtein
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa
import os

model_name = "facebook/wav2vec2-base-960h" # "facebook/wav2vec2-base-960h" or "facebook/wav2vec2-large-960h-lv60-self"
train_data = "./data/L2-Arctic-plus/train_data.json"
test_data = "./data/L2-Arctic-plus/test_data.json"
output_train_data = f"./pipelines/cascaded_asr_llms/asr/results/train/{model_name.split('/')[-1]}.json"
output_test_data = f"./pipelines/cascaded_asr_llms/asr/results/test/{model_name.split('/')[-1]}.json"

os.makedirs(os.path.dirname(output_train_data), exist_ok=True)
os.makedirs(os.path.dirname(output_test_data), exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for data_path in [train_data, test_data]:
    with open(data_path, "r") as f:
        audio_text_pairs = json.load(f)

    audio_text_pairs = audio_text_pairs[:2]
    transcription_results = []

    for pair in tqdm(audio_text_pairs, desc=f"Transcribing {data_path}"):
        audio_path = pair['audio_path']
        text = pair['text']
        annotation_info = pair["annotation_info"]
        mis_exp_sug = pair["mis_exp_sug"]

        audio_input, orig_sr = sf.read(audio_path)
        if orig_sr != 16000:
            audio_input = librosa.resample(audio_input, orig_sr=orig_sr, target_sr=16000)

        input_values = processor(audio_input, return_tensors="pt", padding="longest", sampling_rate=16000).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcribed_text = processor.batch_decode(predicted_ids)[0]

        transcription_results.append({
            "audio_path": audio_path,
            "text": text,
            "transcribed_text": transcribed_text,
            "annotation_info": annotation_info,
            "mis_exp_sug": mis_exp_sug
        })

        print(transcription_results[-1])

    output_path = output_train_data if data_path == train_data else output_test_data
    with open(output_path, "w") as f:
        print(f"Saving results to {output_path}")
        json.dump(transcription_results, f, indent=4, ensure_ascii=False)
