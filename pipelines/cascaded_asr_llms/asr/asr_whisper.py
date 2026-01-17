import json
from tqdm import tqdm
import torch
import jiwer
import Levenshtein
import whisper
import os

# model_name = "large"  # "small" or "medium" or "large"
for model_name in ["small", "medium", "large"]:

    train_data = "./data/L2-Arctic-plus/train_data.json"
    test_data = "./data/L2-Arctic-plus/test_data.json"
    output_train_data = f"./pipelines/cascaded_asr_llms/asr/results/train/whisper-{model_name.split('/')[-1]}.json"
    output_test_data = f"./pipelines/cascaded_asr_llms/asr/results/test/whisper-{model_name.split('/')[-1]}.json"

    model = whisper.load_model(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data_path in [train_data, test_data]:

        with open(data_path, "r") as f:
            audio_text_pairs = json.load(f)
        
        # audio_text_pairs = audio_text_pairs[:2]
        transcription_results = []
        
        for pair in tqdm(audio_text_pairs, desc="Transcribing audio"):
            audio_path = pair["audio_path"]
            text = pair['text']
            annotation_info = pair["annotation_info"]
            mis_exp_sug = pair["mis_exp_sug"]

            result = model.transcribe(audio_path, language="en")
            transcribed_text = result["text"]

            transcription_results.append({
                "audio_path": audio_path,
                "text": text,
                "transcribed_text": transcribed_text,
                "annotation_info": annotation_info,
                "mis_exp_sug": mis_exp_sug
            })

            output_path = output_train_data if data_path == train_data else output_test_data
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                print(f"Saving results to {output_path}")
                json.dump(transcription_results, f, indent=4, ensure_ascii=False)

