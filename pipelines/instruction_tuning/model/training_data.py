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


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")

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
        try:
            item = self.dataset[i]
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

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=128002 if self.tokenizer.pad_token is None else self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
           
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch

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