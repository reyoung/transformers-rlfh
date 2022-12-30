import argparse
import json
import os
import time
import math
from typing import Tuple
import collections

import datasets
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler
import optuna
import numpy

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
               special_token, wandb_enabled=False):
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=200))

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-2)
        scheduler_type = trial.suggest_categorical("scheduler", ["linear", "cosine"])
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1)
        grad_accumulate_steps = trial.suggest_int("grad_accumulate_steps", 1, 4)
        reward_type = trial.suggest_categorical("reward_type", ["last_token", "last_padding"])

        trial_id = trial.number
        if wandb_enabled:
            wandb.init(project="gpt-n-best",
                       name=f"lr-{lr:.2f}-"
                            f"scheduler-{scheduler_type}-warmup-{warmup_ratio:0.2f}-"
                            f"grad-acc-{grad_accumulate_steps}-reward-{reward_type}",
                       reinit=True,
                       config={
                           "lr": lr,
                           "scheduler": scheduler_type,
                           "warmup_ratio": warmup_ratio,
                           "grad_accumulate_steps": grad_accumulate_steps,
                           "reward_type": reward_type
                       })

        model, tokenizer = load_model_and_tokenizer(model_type, special_token)
        collator = BestOfNCollator(tokenizer, special_token=special_token,
                                   pad_token_id=tokenizer.eos_token_id if reward_type == "last_token" else None,
                                   )
        data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size,
                                 num_workers=2, pin_memory=True, shuffle=True)
        model = GPTBestOfN(base=model).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_steps = math.ceil(len(dataset) / batch_size) * epoch // grad_accumulate_steps
        scheduler = get_scheduler(scheduler_type, optimizer, num_warmup_steps=math.ceil(n_steps * warmup_ratio),
                                  num_training_steps=n_steps)
        step_id = 0
        trial_dir = "./trial_{}".format(trial_id)
        os.mkdir(trial_dir)

        times = collections.deque(maxlen=100)

        with open(f"{trial_dir}/log.jsonl", "w") as f:
            json.dump({"lr": lr, "scheduler": scheduler_type, "warmup_ratio": warmup_ratio}, f)
            f.write("\n")
            f.flush()
            optimizer.zero_grad()
            for epoch_id in tqdm.tqdm(range(epoch), desc="epoch"):
                for batch_id, batch in enumerate(tqdm.tqdm(data_loader)):
                    begin = time.time()
                    input_ids = batch[0].to(device, non_blocking=True)
                    best = batch[1].to(device, non_blocking=True)
                    if len(batch) == 2:
                        loss = model(input_ids=input_ids, best=best)
                    else:
                        last_token_pos = batch[2].to(device, non_blocking=True)
                        loss = model(input_ids=input_ids, best=best, last_token_pos=last_token_pos.to(device),
                                     pad_token_id=tokenizer.eos_token_id)
                    loss.backward()
                    if (step_id + 1) % grad_accumulate_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    batch_id += 1

                    if batch_id % log_interval == 0:
                        loss_val = loss.item()
                        trial.report(loss, step_id)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                        step_duration = numpy.mean(times)

                        if wandb_enabled:
                            wandb.log({"loss": loss_val, "step": step_id, "epoch": epoch_id, "batch": batch_id,
                                       "lr": scheduler.get_last_lr()[0], "step_duration": step_duration})
                        json.dump({"epoch": epoch_id, "batch": batch_id, "loss": loss_val,
                                   "lr": scheduler.get_last_lr()}, f)
                        f.write("\n")
                        f.flush()

                    if batch_id % save_interval == 0:
                        state_dict = model.state_dict()
                        torch.save(state_dict, f"{trial_dir}/model_{epoch_id}_{batch_id}.pt")
                    step_id += 1
                    times.append(time.time() - begin)

                state_dict = model.state_dict()
                torch.save(state_dict, f"{trial_dir}/model_{epoch_id}.pt")

        return loss.item()

    study.optimize(objective, n_trials=500)


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
    arg_parser.add_argument("--wandb-token", type=str, default="")
    args = arg_parser.parse_args()

    if len(args.wandb_token) != 0:
        wandb.login(key=args.wandb_token)
        wandb_enabled = True
    else:
        wandb_enabled = False

    device = torch.device(args.device)
    batch_size = args.batch_size
    ds = datasets.load_from_disk(args.data)

    train_main(dataset=ds["train"], model_type=args.model_type, batch_size=batch_size, epoch=args.n_epochs,
               log_interval=args.log_interval, save_interval=args.save_interval, special_token=args.special_token,
               device=device, wandb_enabled=wandb_enabled)


if __name__ == '__main__':
    main()
