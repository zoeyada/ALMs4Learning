import os
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

FILES_TO_PROCESS = [
    # cascaded asr llms
    "./pipelines/cascaded_asr_llms/parse/results/llama/wav2vec2-base-960h.json",
    "./pipelines/cascaded_asr_llms/parse/results/llama/wav2vec2-large-960h-lv60-self.json",
    "./pipelines/cascaded_asr_llms/parse/results/llama/whisper-large.json",
    "./pipelines/cascaded_asr_llms/parse/results/llama/whisper-medium.json",
    "./pipelines/cascaded_asr_llms/parse/results/llama/whisper-small.json",

    "./pipelines/cascaded_asr_llms/parse/results/mistral/wav2vec2-base-960h.json",
    "./pipelines/cascaded_asr_llms/parse/results/mistral/wav2vec2-large-960h-lv60-self.json",
    "./pipelines/cascaded_asr_llms/parse/results/mistral/whisper-large.json",
    "./pipelines/cascaded_asr_llms/parse/results/mistral/whisper-medium.json",
    "./pipelines/cascaded_asr_llms/parse/results/mistral/whisper-small.json",

    # existing ALMs
    "./pipelines/existing_alms/parse/results/gpt4o.json",
    "./pipelines/existing_alms/parse/results/qwen.json",
    "./pipelines/existing_alms/parse/results/qwen2.json",

    # instruction tuning
    "./pipelines/instruction_tuning/parse/results/llama_whisper_large.json"
]

OUTPUT_PATH = "./eval/eval_results.json"


# Mispronunciation Detection Evaluation (MDE)
def calculate_word_level_metrics(data):
    TP = TN = FP = FN = extra_word_count = 0
    all_words = set(word.lower() for word in data.get("text", "").split())
    ground_truth = set((data.get("mis_exp_sug") or {}).keys())
    predicted = set(word.lower() for word in (data.get("speculative_mispronunciations") or {}).keys())

    TP += len(ground_truth.intersection(predicted))
    FP += len(predicted - ground_truth)
    FN += len(ground_truth - predicted)
    TN += len(all_words - ground_truth - predicted)
    extra_word_count += len(predicted - all_words)
    return TP, FP, FN, TN, extra_word_count, len(all_words)


def eval_mde(json_data):
    total_TP = total_FP = total_FN = total_TN = total_extra = total_words = 0
    for entry in json_data:
        TP, FP, FN, TN, extra, words = calculate_word_level_metrics(entry)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN
        total_extra += extra
        total_words += words

    total = total_TP + total_FP + total_FN + total_TN
    accuracy = (total_TP + total_TN) / total if total > 0 else 0
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    extra_ratio = total_extra / total_words if total_words > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Extra Word Ratio": extra_ratio
    }


# Feedback Generation Evaluation (FGE)
def extract_word_issues(data, mode="issue+suggestion"):
    pairs = set()
    if not data:
        return {"no problem"}
    for word, details in data.items():
        for item in details if isinstance(details, list) else []:
            issue = item.get("issue", "")
            suggestion = item.get("suggestion", "")
            pairs.add(f"{word}:{issue} {suggestion}")
    return pairs


def extract_combined_text(data, field):
    texts = []
    for entry in data:
        pairs = extract_word_issues(entry.get(field, {}))
        texts.append(" ".join(pairs))
    return texts


def eval_fge(json_data):
    true_texts = extract_combined_text(json_data, "mis_exp_sug")
    pred_texts = extract_combined_text(json_data, "speculative_mispronunciations")

    bleu_scores, rouge_scores = [], []
    bert_p, bert_r, bert_f1 = [], [], []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for true, pred in zip(true_texts, pred_texts):
        bleu_scores.append(sentence_bleu([true.split()], pred.split(), weights=(0.5, 0.5)))
        rouge_scores.append(rouge.score(true, pred)["rougeL"].fmeasure)
        P, R, F1 = score([pred], [true], lang="en", verbose=False)
        bert_p.append(P.mean().item())
        bert_r.append(R.mean().item())
        bert_f1.append(F1.mean().item())

    n = max(len(true_texts), 1)
    return {
        "BLEU-2": sum(bleu_scores) / n,
        "Rouge-L": sum(rouge_scores) / n,
        "BERTScore Precision": sum(bert_p) / n,
        "BERTScore Recall": sum(bert_r) / n,
        "BERTScore F1": sum(bert_f1) / n,
    }



def detect_pipeline_type(path: str):
    path_lower = path.lower()
    if "cascaded_asr_llms" in path_lower:
        return "cascade_asr_llms"
    elif "instruction_tuning" in path_lower:
        return "instruction_tuning"
    elif "existing_alms" in path_lower or any(x in path_lower for x in ["gpt", "qwen"]):
        return "existing_alms"
    else:
        return "unknown"

def main():
    results = []

    for file_path in FILES_TO_PROCESS:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        mde_metrics = eval_mde(data)
        fge_metrics = eval_fge(data)

        fname = os.path.basename(file_path).replace(".json", "")
        pipeline_type = detect_pipeline_type(file_path)

        if pipeline_type == "existing_alms":
            alm_name = (
                "gpt4o" if "gpt4o" in fname else
                "qwen2" if "qwen2" in fname else
                "qwen" if "qwen" in fname else
                fname
            )
            results.append({
                "pipeline_type": "existing_alms",
                "alm": alm_name,
                "MDE": mde_metrics,
                "FGE": fge_metrics
            })
        else:
            llm = "llama" if "llama" in file_path else "mistral" if "mistral" in file_path else "other"
            asr = "whisper" if "whisper" in fname else "wav2vec2" if "wav2vec2" in fname else "none"
            variant = (
                "base" if "base" in fname else
                "small" if "small" in fname else
                "medium" if "medium" in fname else
                "large" if "large" in fname else
                "none"
            )
            results.append({
                "pipeline_type": pipeline_type,
                "llm": llm,
                "asr": asr,
                "variant": variant,
                "MDE": mde_metrics,
                "FGE": fge_metrics
            })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
