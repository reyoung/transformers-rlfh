from typing import Optional, List

import torch.nn
from transformers_rlfh.models.gpt_best_of_n import GPTBestOfN
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from transformers_rlfh.datasets.best_of_n_collator import BestOfNEvalCollator

__all__ = ['BestOfNScorer']


class BestOfNScorer:
    @staticmethod
    def load_model(pretrained_model: str, path: str, special_token: str, device=None,
                   last_token=False) -> 'BestOfNScorer':
        model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        model = model.transformer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        tokenizer.add_tokens([special_token])
        model.resize_token_embeddings(len(tokenizer))
        model = GPTBestOfN(model)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(torch.device(device))

        return BestOfNScorer(model, special_token, tokenizer,
                             pad_token_id=tokenizer.eos_token_id if last_token else None)

    def __init__(self,
                 model: GPTBestOfN,
                 special_token: str,
                 tokenizer: PreTrainedTokenizerFast,
                 pad_token_id: Optional[int] = None):
        self.collator = BestOfNEvalCollator(tokenizer, special_token, pad_token_id)
        self.bestOfN = model
        self.bestOfN.eval()
        self.pad_token_id = pad_token_id

    def __call__(self, queries: List[str], responses: List[str]) -> List[float]:
        args = self.collator(queries, responses)
        with torch.no_grad():
            reward = self.bestOfN.get_reward(*[item.to(self.bestOfN.device, non_blocking=True) for item in args],
                                             pad_token_id=self.pad_token_id)
            reward = torch.flatten(reward)
            return reward.tolist()
