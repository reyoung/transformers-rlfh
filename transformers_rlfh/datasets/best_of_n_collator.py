from typing import Union, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

__all__ = ['BestOfNCollator', 'BestOfNEvalCollator']

Tokenizer = Union[PreTrainedTokenizerFast, AutoTokenizer]


class BestOfNEvalCollator:
    """
    BestOfN Evaluation Collator.

    :param tokenizer: 分词器，已经添加了special_token
    :param special_token: 特殊的分隔符。用于分割query和response。
        最终输入为 query + special_token + response + special_token + pad_token_id。 如果pad_token_id为None，
        则pad_token_id = special_token
    :param pad_token_id: 如果为None，则pad_token_id = special_token。否则，pad_token_id为另一个ID，模型对输入进行last pooling时
        使用最后一个special_token
    """

    def __init__(self, tokenizer: Tokenizer, special_token: str, pad_token_id: Optional[int] = None):
        if pad_token_id is None:
            pad_id = tokenizer.convert_tokens_to_ids([special_token])[0]
            tokenizer.pad_token_id = pad_id
            tokenizer.pad_token = special_token
            self._return_last_token_pos = False
        else:
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens([pad_token_id])[0]
            tokenizer.pad_token_id = pad_token_id
            self._return_last_token_pos = True
        self._tokenizer = tokenizer
        self.special_token = special_token

    def _collate(self, texts: List[str]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        if not self._return_last_token_pos:
            texts = self._tokenizer(texts, padding=True, return_tensors="pt",
                                    return_attention_mask=False, return_token_type_ids=False)
            input_ids = texts["input_ids"]

            return input_ids,
        else:
            texts = self._tokenizer(texts, padding=False, return_token_type_ids=False,
                                    return_length=True)
            last_token_pos = texts["length"]
            last_token_pos: torch.Tensor = torch.tensor(last_token_pos, dtype=torch.long)
            last_token_pos -= 1
            input_ids = self._tokenizer.pad({"input_ids": texts["input_ids"]}, return_tensors="pt", padding=True,
                                            return_attention_mask=False)["input_ids"]
            return input_ids, last_token_pos

    def __call__(self, queries: List[str], responses: List[str]) -> \
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """

        :param queries: query的列表
        :param responses: response的列表
        :return: (input_ids,) 或者 (input_ids, last_token_pos)。如果设置过 pad_token_id，则会多返回一个last_token_pos
        """
        if len(queries) != len(responses):
            raise ValueError("queries and responses should have the same length")
        texts = ["".join([query, self.special_token, response, self.special_token]) for query, response in
                 zip(queries, responses)]

        return self._collate(texts)


class BestOfNCollator:
    """
    训练GPT Best Of N排序的collator

    """

    def __init__(self, tokenizer: Tokenizer, special_token: str, n_best: int = 4, pad_token_id: Optional[int] = None):
        self._eval_collator = BestOfNEvalCollator(tokenizer, special_token, pad_token_id)
        self._n_best = n_best

    def __call__(self, batch):
        texts = [
            "".join([item["query"].strip(), self._eval_collator.special_token, item[f"sample{i}"].strip(),
                     self._eval_collator.special_token])
            for item in batch for i in range(self._n_best)]
        result = self._eval_collator._collate(texts)
        best = torch.tensor([item["best"] for item in batch], dtype=torch.long)

        if len(result) == 2:
            input_ids, last_token_pos = result
            return input_ids.reshape(-1, self._n_best, input_ids.shape[-1]), best, last_token_pos.reshape(-1,
                                                                                                          self._n_best)
        else:
            return result[0].reshape(-1, self._n_best, result[0].shape[-1]), best
