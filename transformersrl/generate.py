from typing import Callable, Iterator, List, Optional, Union, Iterable, TypeVar, Dict, Generator
from transformersrl.store import SampleStore
from transformersrl.types import PPOSample, ActorCriticOutput
from transformers import GenerationMixin, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import ModelOutput
import itertools
import torch
import dataclasses
from torchtyping import TensorType
import contextlib
from transformersrl.pad import left_pad_sequence
from torch.nn.utils.rnn import pad_sequence

__all__ = ['generate_ppo_samples']

Tokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
T = TypeVar('T')


def chunk_n(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def set_default_generate_kwargs(kwargs: Dict):
    if 'num_beams' not in kwargs:
        kwargs['num_beams'] = 4
    if 'num_return_sequences' not in kwargs:
        kwargs['num_return_sequences'] = 4
    if 'do_sample' not in kwargs:
        kwargs['do_sample'] = False
    if 'early_stopping' not in kwargs:
        kwargs['early_stopping'] = True


@contextlib.contextmanager
def gpt_pad_left_special_tokenizer(tokenizer: Tokenizer):
    pre_pad_token = tokenizer.pad_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yield tokenizer
    tokenizer.pad_token = pre_pad_token
    tokenizer.padding_side = "right"


@dataclasses.dataclass
class GenerateSample:
    query: str
    response: str
    query_ids: TensorType["query_length", torch.long]
    response_ids: TensorType["response_length", torch.long]


def _generate_response_by_generation_model(
        query: Iterator[str],
        tokenizer: Tokenizer,
        generation_model: GenerationMixin,
        generate_batch_size: int,
        max_query_length: int,
        max_response_length: int,
        generate_kwargs: dict,
) -> Generator[GenerateSample, None, None]:
    for query_text in chunk_n(query, generate_batch_size):
        generate_device = generation_model.device
        cpu = torch.device("cpu")
        with gpt_pad_left_special_tokenizer(tokenizer) as pad_left_tokenizer:
            cpu_query_batch = pad_left_tokenizer(query_text, padding=True, add_special_tokens=False,
                                                 max_length=max_query_length, return_tensors='pt')
        input_ids = cpu_query_batch["input_ids"].to(generate_device)
        attention_mask = cpu_query_batch["attention_mask"].to(generate_device)
        eos_id = tokenizer.eos_token_id
        with torch.no_grad():
            generation_output: ModelOutput = generation_model.generate(
                input_ids=input_ids,
                # max_length is the total length of the query and response
                max_length=max_response_length + input_ids.shape[1],
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                **generate_kwargs,
            )
            sequences_ids = generation_output["sequences"][:, input_ids.shape[1]:].to(cpu)
            num_return_sequences: int = generate_kwargs['num_return_sequences']
            sequences = tokenizer.batch_decode(sequences_ids, skip_special_tokens=True)

            for i, query in enumerate(query_text):
                for j in range(num_return_sequences):
                    # un-padding query_ids and response_ids
                    response_ids = sequences_ids[i * num_return_sequences + j]
                    t = (response_ids == eos_id).nonzero().tolist()
                    if len(t) != 0:
                        response_ids = response_ids[:t[0]]

                    query_ids = cpu_query_batch['input_ids'][i]
                    t = (query_ids == eos_id).nonzero().tolist()
                    if len(t) != 0:
                        query_ids = query_ids[t[-1][0] + 1:]

                    yield GenerateSample(
                        query=query,
                        response=sequences[i * num_return_sequences + j],
                        query_ids=query_ids,
                        response_ids=response_ids,
                    )


@dataclasses.dataclass
class ScoredSample:
    sample: GenerateSample
    score: float


def _score_generated_sample(sample: Iterator[GenerateSample],
                            scorer: Callable[[List[str], List[str]], List[float]],
                            scorer_batch_size: int) -> Generator[ScoredSample, None, None]:
    for batch in chunk_n(sample, scorer_batch_size):
        scores = scorer([sample.query for sample in batch], [sample.response for sample in batch])
        for sample, score in zip(batch, scores):
            yield ScoredSample(sample, score)


def _call_ref_model(scored_sample: Iterator[ScoredSample],
                    reference_model: Callable[..., ActorCriticOutput],
                    reference_batch_size: int,
                    eos_token_id: int) -> Generator[PPOSample, None, None]:
    for batch in chunk_n(scored_sample, reference_batch_size):
        query_ids = left_pad_sequence([sample.sample.query_ids for sample in batch], padding_value=eos_token_id)
        response_ids = pad_sequence([sample.sample.response_ids for sample in batch], padding_value=eos_token_id,
                                    batch_first=True)
        max_query_len = query_ids.shape[1]
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = input_ids.not_equal(eos_token_id).long()
        device = reference_model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            reference_model.eval()
            output = reference_model(input_ids=input_ids, attention_mask=attention_mask)
            output.logits = output.logits.to("cpu")
            output.values = output.values.to("cpu")

            for i, sample in enumerate(batch):
                reward = torch.zeros(size=sample.sample.response_ids.shape, dtype=torch.float)
                reward[-1] = sample.score

                yield PPOSample(
                    query=sample.sample.query_ids,
                    response=sample.sample.response_ids,
                    logits=output.logits[i][max_query_len: max_query_len + len(sample.sample.response_ids)],
                    values=output.values[i][max_query_len: max_query_len + len(sample.sample.response_ids)],
                    reward=reward,
                )


def _adjust_reward(ppo_samples: Iterator[PPOSample],
                   adjust_reward: Optional[Callable[[List[PPOSample]], None]],
                   adjust_reward_batch_size: int) -> Generator[PPOSample, None, None]:
    if adjust_reward is None:
        for sample in ppo_samples:
            yield sample
    else:
        for batch in chunk_n(ppo_samples, adjust_reward_batch_size):
            adjust_reward(batch)
            for sample in batch:
                yield sample


def generate_ppo_samples(
        query: Iterator[str],
        tokenizer: Tokenizer,
        scorer: Callable[[List[str], List[str]], List[float]],
        generation_model: GenerationMixin,
        reference_model: Callable[..., ActorCriticOutput],
        max_query_length: int,
        max_response_length: int,
        adjust_reward: Optional[Callable[[List[PPOSample]], None]] = None,
        generate_batch_size: int = 8,
        generate_kwargs: Optional[Dict] = None,
        scorer_batch_size: int = 8,
        reference_batch_size: int = 8,
        adjust_reward_batch_size: int = 8,
) -> SampleStore[PPOSample]:
    """
    生成PPO samples for training

    :param query: query iterator
    :param tokenizer: 分词器
    :param scorer: 评价函数
    :param generation_model: 生成模型
    :param reference_model: 参考模型, 用于计算参考模型的 prob, value
    :param max_query_length: query最大长度
    :param max_response_length: response最大长度
    :param adjust_reward: 调整reward的函数, 用于增加 kl散度约束。可为空
    :param generate_batch_size:  生成过程的batch size
    :param generate_kwargs: 生成过程的参数
    :param scorer_batch_size: 评价过程的batch size
    :param reference_batch_size: 参考模型的batch size
    :param adjust_reward_batch_size: 调整reward的batch size
    :return: 生成的PPO样本
    """
    if generate_kwargs is None:
        generate_kwargs = dict()
    set_default_generate_kwargs(generate_kwargs)

    store = SampleStore[PPOSample]()

    generated_samples = _generate_response_by_generation_model(
        query=query, tokenizer=tokenizer, generation_model=generation_model,
        max_response_length=max_response_length, max_query_length=max_query_length,
        generate_batch_size=generate_batch_size, generate_kwargs=generate_kwargs)
    scored_samples = _score_generated_sample(generated_samples, scorer, scorer_batch_size)
    ppo_samples = _call_ref_model(scored_samples, reference_model, reference_batch_size, tokenizer.eos_token_id)
    for ppo_sample in _adjust_reward(ppo_samples, adjust_reward, adjust_reward_batch_size):
        store.append(ppo_sample)
    return store
