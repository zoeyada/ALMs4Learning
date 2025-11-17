from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import os

from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch

from pipelines.instruction_tuning.model.modalities.base_modality import Modality
from pipelines.instruction_tuning.model.constants import IGNORE_INDEX
from pipelines.instruction_tuning.model.data_tools import encode_chat

import random
from torch.utils.data import IterableDataset

@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


# def _resolve_dataset(path: str) -> HFDataset:
#     if os.path.exists(path):
#         return load_from_disk(path)
#     # else:
#     #     return load_dataset(path, split="train", data_files="*.arrow")

def _resolve_dataset(path: str):
    print(f"🔍 Checking dataset path: {path}")
    print("📁 Exists?", os.path.exists(path))
    print("📄 Files in path:", os.listdir(path) if os.path.exists(path) else "❌ path not found")

    if os.path.exists(os.path.join(path, "dataset_info.json")):
        print("✅ Detected local Hugging Face dataset — using load_from_disk()")
        return load_from_disk(path)
    else:
        print("🌐 Using load_dataset() (remote or arrow mode)")
        return load_dataset(path, split="train", trust_remote_code=True)


class LMMDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
        model_cls
    ):
        super(LMMDataset, self).__init__()
        self.dataset = _resolve_dataset(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.modalities = modalities
        self.model_cls = model_cls

    def __len__(self):
        return len(self.dataset)

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        # print(self.modalities)
        # print(self.dataset[0])
        # item = self.dataset[0]
        # print(encode_chat(item, self.tokenizer, self.modalities))
        # exit()
        try:
            item = self.dataset[i]
            # print(self.model_cls)
            # exit()
            return encode_chat(item, self.tokenizer, self.modalities, self.model_cls)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)


@dataclass
class DataCollatorForSupervisedLMMDataset:
    tokenizer: transformers.PreTrainedTokenizer
    modalities: List[Modality]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ["input_ids", "labels"]
        )
        
        # pad->eos for Llama 9/20 
        
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        # )
        
        # for mistral
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=128002 if self.tokenizer.pad_token is None else self.tokenizer.pad_token_id
        )
        # print(input_ids[:2])
        # exit()
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        
        # # pad->eos for Llama 9/20  
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.eos_token_id),
        # )
        
        # for mistral      
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch

# class CombinedDataset(IterableDataset):
#     def __init__(self, datasets, seed, weights=None):
#         self._seed = seed
#         self._datasets = datasets
#         self._weights = weights
#         n_datasets = len(datasets)
#         if weights is None:
#             self._weights = [1 / n_datasets] * n_datasets
#         self._rng = random.Random(seed)
        
#     def __iter__(self):
#         # The iterator state is initialized here
#         self._dataset_iters = [iter(dataset) for dataset in self._datasets]
#         return self

#     def __next__(self):
#         idx = self._rng.choices(range(len(self._datasets)), weights=self._weights, k=1)[0]
#         # print('chose dataset', idx, 'with weight', self._weights[idx])
#         return next(self._dataset_iters[idx])
    
#     def __len__(self):
#         return sum(len(dataset) for dataset in self._datasets if hasattr(dataset, '__len__'))

# import random
# from torch.utils.data import Dataset

# class CombinedDataset(Dataset):
#     def __init__(self, datasets, seed, weights=None):
#         self._seed = seed
#         self._datasets = datasets
#         self._weights = weights
        
#         n_datasets = len(datasets)
#         if weights is None:
#             self._weights = [1 / n_datasets] * n_datasets
        
#         self._rng = random.Random(seed)
#         self._cumulative_lengths = [0] + list(map(len, datasets))
#         for i in range(1, len(self._cumulative_lengths)):
#             self._cumulative_lengths[i] += self._cumulative_lengths[i-1]

#     def __len__(self):
#         return sum(len(dataset) for dataset in self._datasets)

#     def _map_index_to_dataset(self, idx):
#         for i in range(1, len(self._cumulative_lengths)):
#             if idx < self._cumulative_lengths[i]:
#                 dataset_idx = i - 1
#                 sample_idx = idx - self._cumulative_lengths[dataset_idx]
#                 return dataset_idx, sample_idx

#     def __getitem__(self, idx):
#         dataset_idx, sample_idx = self._map_index_to_dataset(idx)
#         return self._datasets[dataset_idx][sample_idx]

# class CombinedDataset(Dataset):
#     def __init__(self, datasets, seed, weights=None):

#         self.datasets = datasets
#         self.seed = seed
#         self.weights = weights or [1 / len(datasets)] * len(datasets)
#         self.rng = random.Random(seed)  

#         self.current_indices = [0] * len(datasets)
#         self.dataset_lengths = [len(ds) for ds in datasets]
#         self.min_length = min(self.dataset_lengths)

#     def __len__(self):
#         return self.min_length * len(self.datasets)

#     def __getitem__(self, index):

#         dataset_idx = self.rng.choices(
#             range(len(self.datasets)), weights=self.weights, k=1
#         )[0]

#         sample_idx = self.current_indices[dataset_idx]

#         if sample_idx >= self.dataset_lengths[dataset_idx]:
#             raise StopIteration(f"Dataset {dataset_idx} is exhausted.")
        
#         sample = self.datasets[dataset_idx][sample_idx]
#         self.current_indices[dataset_idx] += 1

#         return sample

class CombinedDataset(Dataset):
    def __init__(self, datasets, seed, weights=None):
        self.datasets = datasets
        self.seed = seed
        self.weights = weights or [1 / len(datasets)] * len(datasets)
        self.rng = random.Random(seed)

        self.current_indices = [0] * len(datasets)
        self.dataset_lengths = [len(ds) for ds in datasets]

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, index):
        while True:
            dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]

            sample_idx = self.current_indices[dataset_idx]
            if sample_idx < self.dataset_lengths[dataset_idx]:
                sample = self.datasets[dataset_idx][sample_idx]
                self.current_indices[dataset_idx] += 1
                return sample

            if all(
                idx >= length
                for idx, length in zip(self.current_indices, self.dataset_lengths)
            ):
                raise StopIteration("All datasets are exhausted.")