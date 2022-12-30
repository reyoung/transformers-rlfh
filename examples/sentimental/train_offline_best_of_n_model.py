import argparse
import json
import os

import math
from typing import Tuple

import datasets
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import optuna

from transformersrl.datasets.best_of_n_collator import BestOfNCollator
from transformersrl.models.gpt_best_of_n import GPTBestOfN


def load_model_and_tokenizer(model_type: str, special_token: str) -> Tuple[AutoModel, AutoTokenizer]:
    # get the base model.
    # 我们不能使用 AutoModel.from_pretrained(model_type) 来加载模型。
    # 因为这个模型是一个 GPT2LMHeadModel，而不是 GPT2Model
    model = AutoModelForCausalLM.from_pretrained(model_type)
    # TODO: Fix this hack code
    model = model.transformer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.add_tokens([special_token])
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def train_main(dataset: datasets.Dataset, model_type, device, batch_size, epoch, log_interval, save_interval,
               special_token):
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)

        trial_id = trial.number
        model, tokenizer = load_model_and_tokenizer(model_type, special_token)
        collator = BestOfNCollator(tokenizer, special_token=special_token)
        data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size,
                                 num_workers=2, pin_memory=True)
        model = GPTBestOfN(base=model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_steps = math.ceil(len(dataset) / batch_size) * epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
        step_id = 0
        trial_dir = "./trial_{}".format(trial_id)
        os.mkdir(trial_dir)

        with open(f"{trial_dir}/log.jsonl", "w") as f:
            json.dump({"lr": lr}, f)
            f.write("\n")
            f.flush()

            for epoch_id in tqdm.tqdm(range(epoch), desc="epoch"):
                for batch_id, (query, samples, best) in enumerate(tqdm.tqdm(data_loader)):
                    query = query.to(device, non_blocking=True)
                    samples = samples.to(device, non_blocking=True)
                    best = best.to(device, non_blocking=True)

                    loss = model(query=query, samples=samples, best=best)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_id += 1

                    if batch_id % log_interval == 0:
                        loss = loss.item()
                        trial.report(loss, step_id)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        json.dump({"epoch": epoch_id, "batch": batch_id, "loss": loss,
                                   "lr": optimizer.param_groups[0]["lr"],
                                   "scheduler_lr": scheduler.get_last_lr()}, f)
                        f.write("\n")
                        f.flush()

                    if batch_id % save_interval == 0:
                        state_dict = model.state_dict()
                        torch.save(state_dict, f"{trial_dir}/model_{epoch_id}_{batch_id}.pt")

                    scheduler.step()
                    step_id += 1

        return loss.item()

    study.optimize(objective, n_trials=100)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-type", type=str, default="EleutherAI/gpt-neo-125M")
    arg_parser.add_argument("--batch-size", type=int, default=4)
    arg_parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    arg_parser.add_argument("--data", type=str, default="./data")
    arg_parser.add_argument("--special_token", type=str, default="<|end of req/rsp|>")
    arg_parser.add_argument("--n_epochs", type=int, default=3)
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument("--save_interval", type=int, default=100)
    args = arg_parser.parse_args()

    device = torch.device(args.device)
    batch_size = args.batch_size
    ds = datasets.load_from_disk(args.data)

    train_main(dataset=ds["train"], model_type=args.model_type, batch_size=batch_size, epoch=args.n_epochs,
               log_interval=args.log_interval, save_interval=args.save_interval, special_token=args.special_token,
               device=device)


if __name__ == '__main__':
    main()
