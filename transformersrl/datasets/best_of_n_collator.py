from typing import Union, List

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

__all__ = ['BestOfNCollator']

Tokenizer = Union[PreTrainedTokenizerFast, AutoTokenizer]


class BestOfNCollator:
    """
    训练GPT Best Of N排序的collator

    """

    def __init__(self, tokenizer: Tokenizer, special_token: str, n_best: int = 4):
        pad_id = tokenizer.convert_tokens_to_ids([special_token])[0]
        tokenizer.pad_token_id = pad_id
        tokenizer.pad_token = special_token

        self._tokenizer = tokenizer
        self._special_token = special_token
        self._n_best = n_best

    def __call__(self, batch):
        query = []
        samples: List[List] = [list() for _ in range(self._n_best)]
        for elem in batch:
            query.append(elem["query"].strip() + " " + self._special_token)
            for i in range(self._n_best):
                samples[i].append(elem[f"sample{i}"].strip() + " " + self._special_token)
        query = self._tokenizer(query, padding=True, return_tensors="pt")["input_ids"]
        for i in range(self._n_best):
            samples[i] = self._tokenizer(samples[i], padding=False)["input_ids"]
        max_seq_len = max((len(s) for sample in samples for s in sample))
        for i in range(self._n_best):
            samples[i] = self._tokenizer.pad({"input_ids": samples[i]}, padding="max_length", max_length=max_seq_len)[
                "input_ids"]

        samples = torch.tensor(samples, dtype=torch.long).permute(1, 0, 2)
        return query, samples, torch.tensor([elem["best"] for elem in batch], dtype=torch.long)
