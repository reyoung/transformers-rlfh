import argparse
import json
import os
import sys
from typing import Tuple

import accelerate
import datasets
import math
import optuna
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_scheduler, GPTNeoForCausalLM

from transformers_rlfh.datasets.best_of_n_collator import BestOfNCollator
from transformers_rlfh.models.gpt_best_of_n import GPTBestOfN


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


def train_main(dataset: datasets.Dataset, model_type, batch_size, epoch, log_interval, save_interval,
               special_token, wandb_report_grad_dist_interval, wandb_project, wandb_enabled=False):
    if wandb_report_grad_dist_interval == 0:
        wandb_report_grad_dist_interval = 1
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=100),
                                sampler=optuna.samplers.TPESampler(seed=42),
                                direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 0.01)
        scheduler_type = trial.suggest_categorical("scheduler", ["linear", "cosine"])
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.2)
        grad_accumulate_steps = 1
        reward_type = trial.suggest_categorical("reward_type", ["last_token", "last_padding"])
        max_grad_norm = 1.0
        accelerator = accelerate.Accelerator(gradient_accumulation_steps=grad_accumulate_steps)

        trial_id = trial.number
        config = {
            "lr": lr,
            "scheduler": scheduler_type,
            "warmup_ratio": warmup_ratio,
            "grad_accumulate_steps": grad_accumulate_steps,
            "reward_type": reward_type,
            "trial_id": trial_id,
        }

        model, tokenizer = load_model_and_tokenizer(model_type, special_token)
        collator = BestOfNCollator(tokenizer, special_token=special_token,
                                   pad_token_id=tokenizer.eos_token_id if reward_type == "last_token" else None,
                                   )
        data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=batch_size,
                                 num_workers=4, pin_memory=True, shuffle=True)
        assert isinstance(model, torch.nn.Module)
        model.train()
        model = GPTBestOfN(base=model)

        if wandb_enabled:
            wandb.init(project=wandb_project, config=config,
                       name=f"lr-{lr:.4f}-"
                            f"scheduler-{scheduler_type}-warmup-{warmup_ratio:0.2f}-"
                            f"grad-acc-{grad_accumulate_steps}-reward-{reward_type}",
                       reinit=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        n_steps = math.ceil(len(dataset) / batch_size) * epoch
        scheduler = get_scheduler(scheduler_type, optimizer, num_warmup_steps=math.ceil(n_steps * warmup_ratio),
                                  num_training_steps=n_steps)
        step_id = 1
        trial_dir = "./trial_{}".format(trial_id)
        os.system("rm -r {}".format(trial_dir))
        os.mkdir(trial_dir)

        model, optimizer, scheduler, data_loader = accelerator.prepare(model, optimizer, scheduler, data_loader)
        model: torch.nn.Module = model

        with open(f"{trial_dir}/log.jsonl", "w") as f:
            json.dump(config, f)
            f.write("\n")
            f.flush()
            for epoch_id in tqdm.tqdm(range(epoch), desc="epoch"):
                for batch_id, batch in enumerate(tqdm.tqdm(data_loader)):
                    with accelerator.accumulate(model):
                        input_ids = batch[0].to(accelerator.device, non_blocking=True)
                        best = batch[1].to(accelerator.device, non_blocking=True)
                        if len(batch) == 2:
                            loss = model(input_ids=input_ids, best=best)
                        else:
                            last_token_pos = batch[2].to(accelerator.device, non_blocking=True)
                            loss = model(input_ids=input_ids, best=best, last_token_pos=last_token_pos,
                                         pad_token_id=tokenizer.eos_token_id)
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        batch_id += 1

                        if step_id % log_interval == 0:
                            loss_val = loss.item()
                            trial.report(loss_val, step_id)
                            if math.isnan(loss_val) or trial.should_prune():
                                raise optuna.TrialPruned()

                            if wandb_enabled:
                                stats = {"loss": loss_val, "epoch": epoch_id, "batch": batch_id,
                                         "lr": scheduler.get_last_lr()[0], "step": step_id}

                                if wandb_report_grad_dist_interval > 0 and step_id % (
                                        log_interval * wandb_report_grad_dist_interval) == 0:
                                    for name, param in model.named_parameters():
                                        if param.grad is None:
                                            print(f"param {name} has no grad", file=sys.stderr)
                                        try:
                                            stats[f"grad/{name}"] = wandb.Histogram(param.grad.cpu().float().numpy())
                                        except ValueError:
                                            print(f"failed to report grad dist for {name}", file=sys.stderr)
                                wandb.log(stats)
                            json.dump({"epoch": epoch_id, "batch": batch_id, "loss": loss_val,
                                       "lr": scheduler.get_last_lr()}, f)
                            f.write("\n")
                            f.flush()

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step_id += 1

                accelerator.save_state(f"{trial_dir}/epoch-{epoch_id}/")

        return loss.item()

    study.optimize(objective, n_trials=500)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-type", type=str, default="EleutherAI/gpt-neo-125M")
    arg_parser.add_argument("--batch-size", type=int, default=4)
    arg_parser.add_argument("--data", type=str, default="./data")
    arg_parser.add_argument("--special_token", type=str, default="<|end of req/rsp|>")
    arg_parser.add_argument("--n_epochs", type=int, default=3)
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument("--save_interval", type=int, default=100)
    arg_parser.add_argument("--wandb-token", type=str, default="")
    arg_parser.add_argument("--wandb-project", type=str, default="gpt-best-of-n-a100")
    arg_parser.add_argument("--wandb-report-grad-dist-interval", type=int, default=-1)
    args = arg_parser.parse_args()

    if len(args.wandb_token) != 0:
        wandb.login(key=args.wandb_token)
        wandb_enabled = True
    else:
        wandb_enabled = False

    batch_size = args.batch_size
    ds = datasets.load_from_disk(args.data)

    train_main(dataset=ds["train"], model_type=args.model_type, batch_size=batch_size, epoch=args.n_epochs,
               log_interval=args.log_interval, save_interval=args.save_interval, special_token=args.special_token,
               wandb_enabled=wandb_enabled,
               wandb_report_grad_dist_interval=args.wandb_report_grad_dist_interval,
               wandb_project=args.wandb_project)


if __name__ == '__main__':
    main()
