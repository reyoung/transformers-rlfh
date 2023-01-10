import multiprocessing
import re
from typing import List, Optional

import datasets
import torch.optim
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM

from transformers_rlfh.generate import generate_ppo_samples
from transformers_rlfh.models.actor_critic import ActorCriticLM, ActorCriticOutput
from transformers_rlfh.loss.gae import calculate_gae
from transformers_rlfh.loss.ppo_loss import ppo_lm_loss
from transformers_rlfh.scorers.best_of_n_scorers import BestOfNScorer
from transformers_rlfh.types import PPOBatch
from accelerate import Accelerator


class SplitSentencesAndRemoveShort:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, min_sentence_length: int,
                 end_of_sentences: Optional[re.Pattern] = None, field_name: str = "text",
                 output_name: str = "sentence"):
        if end_of_sentences is None:
            end_of_sentences = re.compile(r'[.!?\n]|\\n', flags=re.M)
        self._end_of_sentences: re.Pattern = end_of_sentences
        self._tokenizer = tokenizer
        self._tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
        self._min_sentence_length = min_sentence_length
        self._field_name = field_name
        self._output_name = output_name

    def __call__(self, batch: LazyBatch):
        texts: List[str] = batch[self._field_name]
        texts_after_split: List[str] = []
        for text in texts:
            texts_after_split.extend(self._end_of_sentences.split(text))
        texts = texts_after_split

        outputs = self._tokenizer(texts, add_special_tokens=False, return_attention_mask=False, padding=False)
        input_ids: List[List[int]] = outputs["input_ids"]
        assert isinstance(input_ids, list)
        assert isinstance(input_ids[0], list)

        sequences = []

        for sentence in input_ids:
            if len(sentence) >= self._min_sentence_length:
                sequences.append(sentence)

        output = {self._output_name: self._tokenizer.batch_decode(sequences)}
        return output


def prepare_wiki_sentence():
    ds = datasets.load_dataset("wikipedia", "20220301.en", keep_in_memory=False)
    nproc = (multiprocessing.cpu_count() - 1) if multiprocessing.cpu_count() > 1 else 1
    ds = ds.map(SplitSentencesAndRemoveShort(AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M"), 10),
                batched=True, remove_columns=ds.column_names['train'],
                num_proc=nproc,
                batch_size=256)
    ds.save_to_disk("wikipedia_20220301.en")


class ScorerWithLogger:
    def __init__(self, scorer):
        self.scorer = scorer

    def __call__(self, quries, responses):
        scores = self.scorer(quries, responses)
        print(list(zip(quries, responses, scores)))
        return scores


def main():
    ds = datasets.load_from_disk("./wikipedia_20220301.en", keep_in_memory=False)
    ds = ds['train']

    data_loader = DataLoader(ds, batch_size=8, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
    model = ActorCriticLM(model)
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    scorer = BestOfNScorer.load_model("EleutherAI/gpt-neo-125M", "/data/tmp2/trial_57/epoch-29/pytorch_model.bin",
                                      special_token="<|endofquery|>", device="cuda:0")

    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer = accelerator.prepare(model, optimizer)

    for batch in data_loader:
        model.eval()
        ppo_samples = generate_ppo_samples(query=batch['sentence'],
                                           tokenizer=tokenizer,
                                           training_model=model,
                                           max_query_length=128,
                                           max_response_length=128,
                                           scorer=scorer,
                                           generate_batch_size=2,
                                           scorer_batch_size=2)
        model.train()

        ppo_loader = ppo_samples.create_data_loader(2, padding_value=tokenizer.eos_token_id)
        ppo_loader = accelerator.prepare(ppo_loader)

        for epoch in range(3):
            for ppo_batch in ppo_loader:
                ppo_batch: PPOBatch = ppo_batch
                model_input = ppo_batch.to_model_input()
                output: ActorCriticOutput = model(input_ids=model_input.input_ids,
                                                  attention_mask=model_input.attention_mask)
                output = output.slice_response(query_length=ppo_batch.query.shape[1])
                result = calculate_gae(value=ppo_batch.values, reward=ppo_batch.rewards)
                advantages, returns = result.advantages, result.returns
                output_dist = torch.distributions.Categorical(logits=output.logits)

                log_probs = output_dist.log_prob(ppo_batch.response)
                values_preds = output.values

                loss = ppo_lm_loss(logprobs=log_probs, values=values_preds,
                                   old_logprobs=ppo_batch.log_probs, old_values=ppo_batch.values,
                                   advantages=advantages, returns=returns,
                                   mask=torch.ne(ppo_batch.response, ppo_batch.padding_value).to(torch.float32))
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                print(loss.item())

            exit(0)


if __name__ == '__main__':
    # prepare_wiki_sentence()
    main()
