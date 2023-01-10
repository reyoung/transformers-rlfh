import json
from typing import Union, List, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader, Dataset

from transformers_rlfh.pad import left_pad_sequence
from transformers_rlfh.types import PPOSample, PPOBatch
from transformers_rlfh.utils import JSONEncoder

__all__ = ['SampleStore']


def ppo_collator(samples: List[PPOSample], padding_value: int, **kwargs) -> PPOBatch:
    return PPOBatch(
        query=left_pad_sequence([sample.query for sample in samples], padding_value=padding_value),
        response=pad_sequence([sample.response for sample in samples], batch_first=True, padding_value=padding_value),
        logits=pad_sequence([sample.logits for sample in samples], batch_first=True, padding_value=0.0),
        values=pad_sequence([sample.values for sample in samples], batch_first=True, padding_value=0.0),
        rewards=pad_sequence([sample.reward for sample in samples], batch_first=True, padding_value=0.0),
        padding_value=padding_value,
    )


g_collators = {
    PPOSample: ppo_collator,
}


class CollatorWithArgs:
    def __init__(self, collator, **kwargs):
        self._collator = collator
        self._kwargs = kwargs

    def __call__(self, samples):
        return self._collator(samples, **self._kwargs)


T = TypeVar('T', covariant=True, bound=Union[PPOSample])


class SampleStore(Dataset[T]):
    def __init__(self, device=torch.device("cpu")):
        self._data = []
        self._device = device
        self._type = None

    def append(self, sample: T):
        if self._type is None:
            self._type = type(sample)
        else:
            if not isinstance(sample, self._type):
                raise TypeError(f"Expected {self._type}, got {type(sample)}")
        self._data.append(sample.to(self._device))

    def __getitem__(self, index) -> T:
        return self._data[index]

    def json_dump(self, fp):
        for item in self._data:
            json.dump(item.as_dict(), fp, cls=JSONEncoder)
            fp.write("\n")

    def __len__(self) -> int:
        return len(self._data)

    def create_data_loader(self,
                           batch_size: int,
                           shuffle: bool = True,
                           pin_memory: bool = torch.cuda.is_available(), **kwargs) -> DataLoader:
        collator = g_collators[self._type]
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                          collate_fn=CollatorWithArgs(collator, **kwargs))
