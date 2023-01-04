from typing import Union, List, Optional

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

__all__ = ['BestOfNCollator']

Tokenizer = Union[PreTrainedTokenizerFast, AutoTokenizer]


class BestOfNCollator:
    """
    训练GPT Best Of N排序的collator

    """

    def __init__(self, tokenizer: Tokenizer, special_token: str, n_best: int = 4, pad_token_id: Optional[int] = None):
        if pad_token_id is None:
            pad_id = tokenizer.convert_tokens_to_ids([special_token])[0]
            tokenizer.pad_token_id = pad_id
            tokenizer.pad_token = special_token
            self._return_last_token_pos = False
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens([pad_token_id])[0]
            tokenizer.pad_token_id = pad_token_id
            self._return_last_token_pos = True
        self._tokenizer = tokenizer
        self._special_token = special_token
        self._n_best = n_best

    def __call__(self, batch):
        try:
            texts = [
                "".join([item["query"].strip(), self._special_token, item[f"sample{i}"].strip(), self._special_token])
                for item in batch for i in range(self._n_best)]

            if not self._return_last_token_pos:
                texts = self._tokenizer(texts, padding=True, return_tensors="pt",
                                        return_attention_mask=False, return_token_type_ids=False)
                input_ids = texts["input_ids"]
            else:
                texts = self._tokenizer(texts, padding=False, return_token_type_ids=False,
                                        return_length=True)
                last_token_pos = texts["length"]
                last_token_pos = torch.tensor(last_token_pos) - 1
                last_token_pos.resize_(len(batch), self._n_best)
                input_ids = self._tokenizer.pad({"input_ids": texts["input_ids"]}, return_tensors="pt", padding=True,
                                                return_attention_mask=False)["input_ids"]

            input_ids.resize_(len(batch), self._n_best, input_ids.shape[1])
            returns = [input_ids, torch.tensor([elem["best"] for elem in batch], dtype=torch.long)]

            if self._return_last_token_pos:
                returns.append(last_token_pos)

            return returns
        except Exception:
            with open("error.log", "a") as f:
                import traceback
                traceback.print_exc(file=f)
            raise
