import os

import accelerate
import datasets
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformers_rlfh.datasets.best_of_n_collator import BestOfNCollator


def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_tokens(["<|end of rsp|>"])
    path = "data"
    ds = datasets.load_from_disk(path)
    data_loader = DataLoader(ds["train"],
                             collate_fn=BestOfNCollator(tokenizer, special_token="<|end of rsp|>", n_best=2,
                                                        pad_token_id=tokenizer.eos_token_id),
                             shuffle=True, batch_size=144, num_workers=6, pin_memory=True)

    for _ in tqdm.tqdm(range(500)):
        for _ in tqdm.tqdm(range(20)):
            for input_ids, n_best, pos in data_loader:
                pass


if __name__ == '__main__':
    main()
