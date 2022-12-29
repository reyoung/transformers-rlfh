import os
from typing import List

import datasets
import torch.optim.optimizer
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformersrl.models.gpt_best_of_n import GPTBestOfN

SPECIAL_TOKEN = "<|end of req rsp|>"


def strip_space_and_add_special_token(batch):
    return {k: [elem.strip() + " " + SPECIAL_TOKEN for elem in batch[k]] for k in
            ["query", "sample0", "sample1", "sample2", "sample3"]}


class Tokenize:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __call__(self, batch):
        return {key: self._tokenizer(batch["query"], padding=True, return_tensors='pt')["input_ids"]
                for key in ["query", "sample0", "sample1", "sample2", "sample3"]}


def data_collator_with_padding(tokenizer):
    def collator(batch):
        samples: List[List] = [list() for _ in range(4)]
        max_sample_length = 0

        for sample in batch:
            for i in range(4):
                samples[i].append(sample[f"sample{i}"])
            max_sample_length = max(max_sample_length, *[len(s[-1]) for s in samples])

        for i, sample in enumerate(samples):
            sample = tokenizer.pad({"input_ids": sample}, padding="max_length", max_length=max_sample_length,
                                   return_tensors="pt")["input_ids"]
            samples[i] = sample

        samples: List[torch.Tensor] = samples

        return {
            "query":
                tokenizer.pad({"input_ids": [sample["query"] for sample in batch]}, padding=True, return_tensors="pt")[
                    "input_ids"],
            "samples": torch.stack(samples).permute(1, 0, 2),
            "best": torch.tensor(list(map(lambda x: x["best"], batch)), dtype=torch.long)
        }

    return collator


def test_train_gpt_best_of_n():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_tokens([SPECIAL_TOKEN])
    special_token_id = len(tokenizer) - 1
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = SPECIAL_TOKEN
    tokenizer.pad_token_id = special_token_id
    path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sentimental", "data")
    ds = datasets.load_from_disk(path)
    ds = ds.map(strip_space_and_add_special_token, batched=True)
    ds = ds.map(Tokenize(tokenizer), batched=True, batch_size=8)

    dataloader = DataLoader(dataset=ds["train"], collate_fn=data_collator_with_padding(tokenizer), batch_size=2)

    model = GPTBestOfN(base=model)
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(1):
        for databatch in tqdm.tqdm(dataloader):
            loss = model(query=databatch["query"].to(device),
                         samples=databatch["samples"].to(device),
                         best=databatch["best"].to(device), )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
            break
