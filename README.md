
# Unlocking Large Audio-Language Models for Interactive Language Learning

![Pipeline Overview](/image/fig_pipeline.jpg)

This repository contains code and analysis for the paper [unlocking Large Audio-Language Models for Interactive Language Learning](https://arxiv.org/pdf/2310.09505#page=10.70).

We investigate how **Large Audio-Language Models (ALMs)** can be adapted for **L2 pronunciation training**, focusing on mispronunciation detection and human-friendly feedback generation.  The system covers three components:

1. **Cascaded ASR + LLM**
2. **Existing Audio Language Models (ALMs)**
3. **Instruction-Tuning Pipeline (Two-Stage)**

All experiments were conducted on **NVIDIA A40 GPUs (46GB)** using strictly controlled prompts and evaluation protocols.

## 🛠️ Environment & Dependencies

```bash
git clone <your-git-remote> ALMs4Learning
cd ALMs4Learning

conda env create -f environment_base.yml
conda env create -f environment_instruct_tuning.yml
conda activate ALMs_base
```

## 📁 Project Layout
```bash
ALMs4Learning/
├── data/
│   ├── L2-Arctic-plus/           # L2-ARCTIC raw + augmented annotations
│   └── training_datasets/        # HuggingFace datasets for pretrain/finetune
├── pipelines/
│   ├── cascaded_asr_llms/        # ASR front-ends, LLM back-ends, parsing
│   ├── existing_alms/            # GPT-4o, Qwen Audio, Qwen2 Audio baselines
│   └── instruction_tuning/       # Model configs, training, inference, parsing
├── eval/                         # Unified evaluation script + results
├── image/
├── environment_base.yml
└── environment_instruct_tuning.yml
```

## 🎙️ Data Preparation
### 1. Download L2-ARCTIC 

Request access at: https://psi.engr.tamu.edu/l2-arctic-corpus/

After submitting your information you will receive Google Drive links.
Download "L2-ARCTIC-V5.0 (everything packed)", unpack it, and place:
```bash
./data/L2-Arctic-plus/l2arctic_release_v5.0/
```

### 2. L2-Arctic-plus Annotation Files

This repository includes augmented prompts + GPT-4o annotations:
```bash
./data/L2-Arctic-plus/train_data.json   # (1,699 entries)
./data/L2-Arctic-plus/test_data.json    # (900  entries)
```
- train_data.json → instruction tuning
- test_data.json → evaluation & baselines

### 3. Build Two-Stage Instruction-Tuning Datasets

Pretraining Dataset:
```bash
python data/training_datasets/build/build_pretrain_dataset.py \
    --output_folder ./data/training_datasets/pretrain \
    --max_examples 200000
```

Finetuning Dataset:
```bash
python data/training_datasets/build/build_finetune_dataset.py \
    --output_folder ./data/training_datasets/finetune \
    --num_proc 10
```

Generated HuggingFace datasets:
```bash
./data/training_datasets/pretrain/
./data/training_datasets/finetune/
```

## 🔁 Pipelines
### 1. Cascaded ASR + LLMs

#### ASR Transcription
Use:
```bash
python ./pipelines/cascaded_asr_llms/asr/asr_whisper.py
python ./pipelines/cascaded_asr_llms/asr/wav2vec2.py
```
Each script loads `train_data.json` and `test_data.json`, runs ASR using Whisper or Wav2Vec2, and writes outputs to:

```bash
./pipelines/cascaded_asr_llms/asr/results/{train,test}/<asr>.json
```

#### LLM Feedback Generation
Use:
```bash
python ./pipelines/cascaded_asr_llms/llms/generate.py
```
This script:

loads ASR outputs

- normalizes transcript text
- formats prompts
- sends them to Mistral/Llama models

Results saved to:
```bash
./pipelines/cascaded_asr_llms/llms/results/<llm>/<asr>.json
```
#### Parsing LLM Feedback

Use:
```bash
python ./pipelines/cascaded_asr_llms/parse/parse.py
```

Outputs placed under:
```bash
./pipelines/cascaded_asr_llms/parse/results/<llm>/<asr>.json
```

All outputs become normalized dictionaries:
```bash
{
  "word": [
    {"issue": "...", "suggestion": "..."}
  ]
}
```


### 2. Existing Audio Language Models (ALMs)

This repository includes three strong audio-language model baselines:

- **GPT-4o Audio**
- **Qwen Audio**
- **Qwen2 Audio**

**GPT-4o Audio** communicates with the OpenAI `gpt-4o-audio-preview` endpoint using paired reference–utterance audio inputs:

```bash
python ./pipelines/existing_alms/alms/gpt4o_audio.py
```
**Note:** Set your OpenAI API key in `gpt4o_audio.py`.

Qwen Audio and Qwen2 Audio run fully locally via HuggingFace implementations:
```bash
python ./pipelines/existing_alms/alms/qwen_audio.py
python ./pipelines/existing_alms/alms/qwen2_audio.py
```

All raw outputs are saved under:
```bash
./pipelines/existing_alms/alms/results/
```

They can be converted into structured formats using:
```bash
python ./pipelines/existing_alms/parse/parse.py
```

Parsed outputs are stored in:
```bash
./pipelines/existing_alms/parse/results/
```

### 3. Instruction-Tuning Pipeline (Two-Stage)
**Stage 1** learns a universal audio-to-text projector, while **Stage 2** adapts the LLM to L2 pronunciation feedback tasks.

#### Stage 1: Projector Pretraining
Pretraining is launched using the following script:

```bash
conda activate ALMs_instruct_tuning
bash ./pipelines/instruction_tuning/training/pretraining.sh
```

This script is a DeepSpeed wrapper around `train_model.py` and configures:
- LLM backbone
- acoustic encoder (Whisper / Wav2Vec2)
- modality builder
- per-device batch size
- gradient accumulation steps
- learning-rate scheduler

**Note:** Set your configures in `pretraining.sh`

Checkpoints are saved under:
```bash
pipelines/instruction_tuning/checkpoints/pretraining/{llm}_{asr}_{modality}/
```

#### Stage 2: Task Finetuning

Finetuning is launched via:
```bash
bash ./pipelines/instruction_tuning/training/finetuning.sh
```
**Note:** Set your configures in `finetuning.sh`
The script loads the pretrained projector from:
```bash
.../pretraining/.../non_lora_trainables.bin
```

LoRA is enabled during finetuning, using the dataset located at:
```bash
./data/training_datasets/finetune/
```

Finetuned checkpoints are saved under:
```bash
./pipelines/instruction_tuning/checkpoints/finetuning/{llm}_{asr}_{modality}/
```

#### Inference
Run inference with:
```bash
python ./pipelines/instruction_tuning/inference/inference.py
```

This loads a selected LoRA checkpoint and writes predictions to:
```bash
./pipelines/instruction_tuning/inference/results/{llm}_{asr}_{modality}.json
```

#### Parsing Instruction-Tuning Outputs
Normalize and structure inference outputs using:
```bash
python ./pipelines/instruction_tuning/parse/parse.py
```

Parsed results are saved to:
```bash
./pipelines/instruction_tuning/parse/results/{llm}_{asr}_{modality}.json
```

### 📊 Unified Evaluation

Produces 8 metrics across two groups:

Mispronunciation Detection Evaluation (MDE)
- Accuracy
- Precision
- Recall
- F1
- Extra Word Ratio

Feedback Generation Evaluation (FGE)
- BLEU-2
- ROUGE-L
- BERTScore

Run:
```bash
python eval/eval.py
```

Produces:
```bash
./eval/eval_results.json
```

Format:
```bash
[
  {
    "pipeline_type": "cascaded_asr_llms",
    "llm": "llama",
    "asr": "whisper",
    "variant": "large",
    "MDE": {...},
    "FGE": {...}
  },
  {
    "pipeline_type": "existing_alms",
    "alm": "gpt4o",
    "MDE": {...},
    "FGE": {...}
  }
]
```
