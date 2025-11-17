from typing import Dict, List, Any, Union, Optional
from collections import Counter
from functools import cache
import contextlib
import tempfile
import shutil
import random
import subprocess
import json
import re
import io
import os

import torch
import requests
import transformers
import numpy as np
from datasets import load_dataset, Dataset
from PIL import Image

from pipelines.instruction_tuning.model.constants import IGNORE_INDEX

def encode_chat(
    item: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    modalities: List["Modality"],
    model_cls
) -> Dict:
    messages = list(item["messages"])
    
    if (model_cls == "LlamaLMMForCausalLM"):
        chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

        token_to_modality = {m.token: m for m in modalities}

        # print("modalities", modalities)
        # modalities [<multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7f5b25c91420>]
        
        modality_token_counts = Counter()
        instruct_pattern = r"(<\|start_header_id\|>user<\|end_header_id\|>[\s\S]*?<\|eot_id\|>)"
        pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

        chat_part = re.split(instruct_pattern, chat_as_string)
        input_ids = []
        labels = []
        
        # print("chat_part:", chat_part)
        
        for part in chat_part:
            if "<|start_header_id|>user<|end_header_id|>" in part:
                is_instruction = True
            else:
                is_instruction = False
            for subpart in re.split(pattern, part):
                if not subpart:
                    continue
                if subpart in token_to_modality:
                    assert (
                        is_instruction
                    ), "There should be no modality tokens outside of instructions"
                    m = token_to_modality[subpart]
                    
                    modality_token_counts[m.name] += 1
                    input_ids.extend([m.token_idx] * m.token_width)
                    labels.extend([IGNORE_INDEX] * m.token_width)
                    
                elif is_instruction:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    input_ids.extend(part_ids)
                    labels.extend([IGNORE_INDEX] * len(part_ids))
                else:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    input_ids.extend(part_ids)
                    labels.extend(part_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )
        for m in modalities:
            data_dict[m.name] = m.preprocess_rows([item])[0]
        # # print("LlamaLMMForCausalLM")
        # # exit()
        
        
        # chat_as_string = ""
        # for message in messages:
        #     if message["role"] == "user":
        #         chat_as_string += message["content"] + " "
        #     elif message["role"] in ["system", "assistant"]:
        #         chat_as_string += message["content"] + " "
        
        # # print("chat_as_string", chat_as_string)
        
        # token_to_modality = {m.token: m for m in modalities}
        # modality_token_counts = Counter()

        # pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
        # chat_part = re.split(pattern, chat_as_string)

        # # print("chat_part", chat_part)
        # # exit()
        # input_ids = []
        # labels = []

        # for part in chat_part:
        #     if not part:
        #         continue
        #     if part in token_to_modality:
        #         m = token_to_modality[part]
        #         modality_token_counts[m.name] += 1
        #         input_ids.extend([m.token_idx] * m.token_width)
        #         labels.extend([IGNORE_INDEX] * m.token_width)
        #     else:
        #         part_ids = tokenizer(part, add_special_tokens=False).input_ids
        #         input_ids.extend(part_ids)
        #         labels.extend(part_ids)

        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # labels = torch.tensor(labels, dtype=torch.long)
        
        # # print("input_ids", input_ids, "labels", labels)
        # # exit()
        
        # data_dict = {
        #     "input_ids": input_ids,
        #     "labels": labels,
        # }
        # for modality in modalities:
        #     # if not modality:
        #     #     continue
        #     # print("modality.name:", modality.name)
        #     # exit()
        #     data_dict[modality.name] = modality.preprocess_rows([item])[0]
        #     # print(data_dict)
        #     # exit()
            
    if (model_cls == "MistralLMMForCausalLM"):
        # print("data_tool, MistralLMMForCausalLM")
        # exit()
        chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

        token_to_modality = {m.token: m for m in modalities}

        # print("modalities", modalities)
        # modalities [<multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7f5b25c91420>]
        
        modality_token_counts = Counter()
        instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
        pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

        chat_part = re.split(instruct_pattern, chat_as_string)
        input_ids = []
        labels = []
        
        # print("chat_part:", chat_part)
        # exit()
        
        for part in chat_part:
            if "[INST]" in part:
                is_instruction = True
            else:
                is_instruction = False
            for subpart in re.split(pattern, part):
                if not subpart:
                    continue
                if subpart in token_to_modality:
                    assert (
                        is_instruction
                    ), "There should be no modality tokens outside of instructions"
                    m = token_to_modality[subpart]
                    # print("m", m, m.name) # m <multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7fb774e00e20> audio_whisper
                    
                    modality_token_counts[m.name] += 1
                    input_ids.extend([m.token_idx] * m.token_width)
                    labels.extend([IGNORE_INDEX] * m.token_width)
                    
                    # print("token_width:", m.token_width) # 10
                elif is_instruction:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    input_ids.extend(part_ids)
                    labels.extend([IGNORE_INDEX] * len(part_ids))
                else:
                    part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
                    input_ids.extend(part_ids)
                    labels.extend(part_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
        )
        for m in modalities:
            data_dict[m.name] = m.preprocess_rows([item])[0]
            
        # print("data_dict:", data_dict)

    return data_dict

# Change encode_chat for Llama 11/4
# def encode_chat(
#     item: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     modalities: List["Modality"],
# ) -> Dict:
#     messages = list(item["messages"])
    
#     chat_as_string = ""
#     for message in messages:
#         if message["role"] == "user":
#             chat_as_string += message["content"] + " "
#         elif message["role"] in ["system", "assistant"]:
#             chat_as_string += message["content"] + " "
    
#     # print("chat_as_string", chat_as_string)
    
#     token_to_modality = {m.token: m for m in modalities}
#     modality_token_counts = Counter()

#     pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"
#     chat_part = re.split(pattern, chat_as_string)

#     # print("chat_part", chat_part)
#     # exit()
#     input_ids = []
#     labels = []

#     for part in chat_part:
#         if not part:
#             continue
#         if part in token_to_modality:
#             m = token_to_modality[part]
#             modality_token_counts[m.name] += 1
#             input_ids.extend([m.token_idx] * m.token_width)
#             labels.extend([IGNORE_INDEX] * m.token_width)
#         else:
#             part_ids = tokenizer(part, add_special_tokens=False).input_ids
#             input_ids.extend(part_ids)
#             labels.extend(part_ids)

#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)
    
#     # print("input_ids", input_ids, "labels", labels)
#     # exit()
    
#     data_dict = {
#         "input_ids": input_ids,
#         "labels": labels,
#     }
#     for modality in modalities:
#         # if not modality:
#         #     continue
#         # print("modality.name:", modality.name)
#         # exit()
#         data_dict[modality.name] = modality.preprocess_rows([item])[0]
#         # print(data_dict)
#         # exit()

#     return data_dict

# # Change encode_chat for Llama 9/24
# """
# chat_part: 
# ['<|begin_of_text|>', 
# '<|start_header_id|>user<|end_header_id|>\n\nCould you transcribe <speech> for me?<|eot_id|>', 
# '<|start_header_id|>assistant<|end_header_id|>\n\nThe school is surrounded by farms and wind turbines.<|eot_id|>']
# """
# def encode_chat(
#     item: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     modalities: List["Modality"],
# ) -> Dict:
#     messages = list(item["messages"])
#     chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

#     token_to_modality = {m.token: m for m in modalities}

#     # print("modalities", modalities)
#     # modalities [<multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7f5b25c91420>]
    
#     modality_token_counts = Counter()
#     instruct_pattern = r"(<\|start_header_id\|>user<\|end_header_id\|>[\s\S]*?<\|eot_id\|>)"
#     pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

#     chat_part = re.split(instruct_pattern, chat_as_string)
#     input_ids = []
#     labels = []
    
#     # print("chat_part:", chat_part)
    
#     for part in chat_part:
#         if "<|start_header_id|>user<|end_header_id|>" in part:
#             is_instruction = True
#         else:
#             is_instruction = False
#         for subpart in re.split(pattern, part):
#             if not subpart:
#                 continue
#             if subpart in token_to_modality:
#                 assert (
#                     is_instruction
#                 ), "There should be no modality tokens outside of instructions"
#                 m = token_to_modality[subpart]
                
#                 modality_token_counts[m.name] += 1
#                 input_ids.extend([m.token_idx] * m.token_width)
#                 labels.extend([IGNORE_INDEX] * m.token_width)
                
#             elif is_instruction:
#                 part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
#                 input_ids.extend(part_ids)
#                 labels.extend([IGNORE_INDEX] * len(part_ids))
#             else:
#                 part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
#                 input_ids.extend(part_ids)
#                 labels.extend(part_ids)

#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)

#     data_dict = dict(
#         input_ids=input_ids,
#         labels=labels,
#     )
#     for m in modalities:
#         data_dict[m.name] = m.preprocess_rows([item])[0]
        
#     return data_dict

# Change encode_chat for Llama 9/20
# Annotated in 9.24, new one above

# def encode_chat(
#     item: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     modalities: List["Modality"],
# ) -> Dict:
#     messages = list(item["messages"])
#     chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

#     token_to_modality = {m.token: m for m in modalities}

#     # Define patterns to detect instructions and modality tokens
#     instruction_start_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>"
#     instruction_end_pattern = r"<\|eot_id\|>"
#     modality_pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

#     # Split the chat into instruction and non-instruction parts
#     chat_parts = re.split(f"({instruction_start_pattern}|{instruction_end_pattern})", chat_as_string)
    
#     print("chat_parts:", chat_parts, "\n")
    
#     input_ids = []
#     labels = []

#     is_instruction = False  # Flag to track if we are in an instruction block
    
#     for part in chat_parts:
#         # Detect the beginning of an instruction block
#         if re.match(instruction_start_pattern, part):
#             is_instruction = True
#             continue
#         # Detect the end of an instruction block
#         elif re.match(instruction_end_pattern, part):
#             is_instruction = False
#             continue

#         # Process the content inside the instruction
#         for subpart in re.split(modality_pattern, part):
#             if not subpart:
#                 continue
            
#             # Handle modality tokens
#             if subpart in token_to_modality:
#                 assert is_instruction, "There should be no modality tokens outside of instructions"
#                 m = token_to_modality[subpart]
#                 input_ids.extend([m.token_idx] * m.token_width)
#                 labels.extend([IGNORE_INDEX] * m.token_width)
#             else:
#                 # Tokenize based on whether it's instruction or regular text
#                 part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
#                 input_ids.extend(part_ids)
#                 if is_instruction:
#                     labels.extend([IGNORE_INDEX] * len(part_ids))  # Mask instructions
#                 else:
#                     labels.extend(part_ids)  # Regular content gets labels

#     # Convert lists to tensors
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)

#     # Create the data dictionary with input_ids and labels
#     data_dict = dict(input_ids=input_ids, labels=labels)

#     # Add modality-specific preprocessed data
#     for m in modalities:
#         data_dict[m.name] = m.preprocess_rows([item])[0]

#     # print("data_dict:", data_dict, "\n")

#     return data_dict

# def encode_chat(
#     item: Dict,
#     tokenizer: transformers.PreTrainedTokenizer,
#     modalities: List["Modality"],
# ) -> Dict:
#     messages = list(item["messages"])
#     chat_as_string = tokenizer.apply_chat_template(messages, tokenize=False)

#     token_to_modality = {m.token: m for m in modalities}

#     # print("modalities", modalities)
#     # modalities [<multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7f5b25c91420>]
    
#     modality_token_counts = Counter()
#     instruct_pattern = r"(\[INST\][\s\S]*?\[\/INST\])"
#     pattern = "(" + "|".join(re.escape(m.token) for m in modalities) + ")"

#     chat_part = re.split(instruct_pattern, chat_as_string)
#     input_ids = []
#     labels = []
    
#     # print("chat_part:", chat_part)
#     # exit()
    
#     for part in chat_part:
#         if "[INST]" in part:
#             is_instruction = True
#         else:
#             is_instruction = False
#         for subpart in re.split(pattern, part):
#             if not subpart:
#                 continue
#             if subpart in token_to_modality:
#                 assert (
#                     is_instruction
#                 ), "There should be no modality tokens outside of instructions"
#                 m = token_to_modality[subpart]
#                 # print("m", m, m.name) # m <multi_token.modalities.audio_whisper.WhisperAudioModality object at 0x7fb774e00e20> audio_whisper
                
#                 modality_token_counts[m.name] += 1
#                 input_ids.extend([m.token_idx] * m.token_width)
#                 labels.extend([IGNORE_INDEX] * m.token_width)
                
#                 # print("token_width:", m.token_width) # 10
#             elif is_instruction:
#                 part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
#                 input_ids.extend(part_ids)
#                 labels.extend([IGNORE_INDEX] * len(part_ids))
#             else:
#                 part_ids = tokenizer(subpart, add_special_tokens=False).input_ids
#                 input_ids.extend(part_ids)
#                 labels.extend(part_ids)

#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     labels = torch.tensor(labels, dtype=torch.long)

#     data_dict = dict(
#         input_ids=input_ids,
#         labels=labels,
#     )
#     for m in modalities:
#         data_dict[m.name] = m.preprocess_rows([item])[0]
        
#     # print("data_dict:", data_dict)
#     return data_dict


def load_image(value: Any) -> Image.Image:
    img = None
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            response = requests.get(value)
            img = Image.open(io.BytesIO(response.content))
        elif os.path.exists(value):
            img = Image.open(value)
    elif isinstance(value, Image.Image):
        img = value
    if img is None:
        raise ValueError(f"Could not load image from {value}")
    img = img.convert("RGB")
    return img


@contextlib.contextmanager
def with_local_files(fn_or_urls: List[Any]):
    local_fns = []
    fps = []
    for fn_or_url in fn_or_urls:
        if isinstance(fn_or_url, Image.Image):
            fp = tempfile.NamedTemporaryFile(suffix=".png", mode="wb")
            fn_or_url.convert("RGB").save(fp)
            fps.append(fp)
            local_fns.append(fp.name)
        elif fn_or_url.startswith("http://") or fn_or_url.startswith("https://"):
            suffix = os.path.splitext(fn_or_url)[-1]
            with requests.get(fn_or_url, stream=True) as r:
                fp = tempfile.NamedTemporaryFile(suffix=suffix, mode="wb")
                shutil.copyfileobj(r.raw, fp)
                fps.append(fp)
                local_fns.append(fp.name)
        else:
            local_fns.append(fn_or_url)
    try:
        yield local_fns
    finally:
        for fp in fps:
            fp.close()


@cache
def _get_dataset(dataset_args: str) -> Dataset:
    return load_dataset(**json.loads(dataset_args))


def get_dataset_cached(dataset_args: Dict) -> Dataset:
    return _get_dataset(json.dumps(dataset_args))


def load_audio(input_: Union[Dict, str], target_sampling_rate: int = None, time: int = 12) -> Dict:
    import soundfile as sf
    import librosa
    import numpy as np

    if isinstance(input_, dict) and "array" in input_ and "sampling_rate" in input_:
        array = input_["array"]
        sampling_rate = input_["sampling_rate"]
    elif isinstance(input_, dict) and "dataset_args" in input_:
        item = get_dataset_cached(input_["dataset_args"])[input_["idx"]]
        array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
    elif isinstance(input_, dict) and "path" in input_:
        with with_local_files([input_["path"]]) as local_fns:
            array, sampling_rate = sf.read(local_fns[0])
    elif isinstance(input_, dict) and "audio_path" in input_:
        with with_local_files([input_["audio_path"]]) as local_fns:
            array, sampling_rate = sf.read(local_fns[0])
    elif isinstance(input_, str):
        with with_local_files([input_]) as local_fns:
            array, sampling_rate = sf.read(local_fns[0])
    else:
        raise ValueError(f"Could not load audio from {input_}")

    if array.ndim == 2:
        array = array.mean(axis=1)
        
    if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
        array = librosa.resample(
            array, orig_sr=sampling_rate, target_sr=target_sampling_rate
        )
        sampling_rate = target_sampling_rate

    # padding
    target_length = target_sampling_rate * time 
    if len(array) < target_length:
        padding = np.zeros(target_length - len(array))
        array = np.concatenate([array, padding])
    else:
        array = array[:target_length]
        
    # print("load_audio_array:", list(array))

    return {"array": list(array), "sampling_rate": sampling_rate}


def _download_yt_video(url: str) -> str:
    from pytube import YouTube

    youtube = YouTube(url)
    video = youtube.streams.first()

    fn = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))
    file_path = video.download(output_path=tempfile.gettempdir(), filename=fn)

    return file_path


def _read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def _sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def load_video(
    input_: str,
    frames: int = 8,
    frame_sample_rate: int = 1,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> np.ndarray:
    import av

    delete_file = False

    if isinstance(input_, dict) and "youtube.com" and input_.get("url", ""):
        file_path = _download_yt_video(input_["url"])
        delete_file = True
        # start_time = input_.get("start_time", None)
        # end_time = input_.get("end_time", None)
    elif isinstance(input_, str) and "youtube.com" in input_:
        file_path = _download_yt_video(input_)
        delete_file = True
    elif isinstance(input_, str):
        file_path = input_
    else:
        raise ValueError(f"Could not load video from {input_}")

    if start_time is not None or end_time is not None:
        start_time = start_time if start_time is not None else 0
        end_time = end_time if end_time is not None else "end"
        trim_file_path = f"{file_path.rsplit('.', 1)[0]}_trim.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                file_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-c",
                "copy",
                trim_file_path,
            ]
        )
        file_path = trim_file_path

    container = av.open(file_path)
    indices = _sample_frame_indices(
        clip_len=frames,
        frame_sample_rate=frame_sample_rate,
        seg_len=container.streams.video[0].frames,
    )
    video = _read_video_pyav(container, indices)

    if delete_file:
        os.remove(file_path)

    return video
