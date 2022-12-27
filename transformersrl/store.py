import torch
from transformersrl.types import PPOSample, PPOBatch, QuerySample, QueryBatch
from torch.utils.data.dataloader import DataLoader, Dataset
from typing import Union, List
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast

__all__ = ['SampleStore']

Sample = Union[PPOSample, QuerySample]
Batch = Union[PPOBatch, QueryBatch]


def ppo_collator(samples: List[PPOSample], pad_token: int, **kwargs) -> PPOBatch:
    return PPOBatch(
        query=pad_sequence([sample.query for sample in samples], batch_first=True, padding_value=pad_token),
        response=pad_sequence([sample.response for sample in samples], batch_first=True, padding_value=pad_token),
        probs=pad_sequence([sample.probs for sample in samples], batch_first=True, padding_value=0.0),
        values=pad_sequence([sample.values for sample in samples], batch_first=True, padding_value=0.0),
        rewards=pad_sequence([sample.reward for sample in samples], batch_first=True, padding_value=0.0),
        pad_token=pad_token,
    )


def query_collator(samples: List[QuerySample], tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   max_length: int,
                   **kwargs) -> QueryBatch:
    collator = DataCollatorWithPadding(tokenizer, max_length=max_length, **kwargs)
    batch = collator([{"text": sample.text} for sample in samples])
    return QueryBatch(input_ids=batch["text"], pad_id=tokenizer.pad_token_id)


g_collators = {
    PPOSample: ppo_collator,
    QuerySample: query_collator,
}


class CollatorWithArgs:
    def __init__(self, collator, **kwargs):
        self._collator = collator
        self._kwargs = kwargs

    def __call__(self, samples):
        return self._collator(samples, **self._kwargs)


class SampleStore(Dataset[Sample]):
    def __init__(self, device=torch.device("cpu")):
        self._data = []
        self._device = device
        self._type = None

    def append(self, sample: Sample):
        if self._type is None:
            self._type = type(sample)
        else:
            if not isinstance(sample, self._type):
                raise TypeError(f"Expected {self._type}, got {type(sample)}")
        self._data.append(sample.to(self._device))

    def __getitem__(self, index) -> Sample:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)

    def create_data_loader(self,
                           batch_size: int,
                           shuffle: bool = True,
                           pin_memory: bool = torch.cuda.is_available()) -> DataLoader:
        collator = g_collators[self._type]
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                          collate_fn=CollatorWithArgs(collator, pad_token=pad_token))
