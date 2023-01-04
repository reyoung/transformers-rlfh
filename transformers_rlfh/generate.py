from typing import Callable, Iterator, List, Optional, Union, Iterable, TypeVar, Dict, Generator
from transformers_rlfh.store import SampleStore
from transformers_rlfh.types import PPOSample, ActorCriticOutput
from transformers import GenerationMixin, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.utils import ModelOutput
import itertools
import torch
import dataclasses
from torchtyping import TensorType
import contextlib
from transformers_rlfh.pad import left_pad_sequence
from torch.nn.utils.rnn import pad_sequence

__all__ = ['generate_ppo_samples']

Tokenizer = Union[PreTrainedTokenizerFast, PreTrainedTokenizer]
T = TypeVar('T')


def chunk_n(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    """
    chunk_n 把一个iterable转换成 iterable of batched samples。n为最大的batch size
    :param iterable:
    :param n:
    :return:
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


@contextlib.contextmanager
def gpt_pad_left_special_tokenizer(tokenizer: Tokenizer):
    """
    GPT left padding的tokenizer的context manager。
    """
    if tokenizer.verbose:
        tokenizer.verbose = False
        verbose_disabled = True
    else:
        verbose_disabled = False
    # save the original pad token，if original pad token is None, we also disable verbose to supress warning
    pre_pad_token = tokenizer.pad_token
    if verbose_disabled:
        tokenizer.verbose = True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    yield tokenizer
    tokenizer.pad_token = pre_pad_token
    tokenizer.padding_side = "right"


@dataclasses.dataclass
class GenerateSample:
    """
    Generation模型生成的样本
    """
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
    """
    用generation model生成response
    """
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
                        response_ids = response_ids[:t[0][0]]

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
    """
    评分后的样本
    """
    sample: GenerateSample
    score: float


def _score_generated_sample(sample: Iterator[GenerateSample],
                            scorer: Callable[[List[str], List[str]], List[float]],
                            scorer_batch_size: int) -> Generator[ScoredSample, None, None]:
    """
    评分生成的样本
    """
    for batch in chunk_n(sample, scorer_batch_size):
        scores = scorer([sample.query for sample in batch], [sample.response for sample in batch])
        for sample, score in zip(batch, scores):
            yield ScoredSample(sample, score)


def _call_training_model(scored_sample: Iterator[ScoredSample],
                         training_model: Callable[..., ActorCriticOutput],
                         training_model_eval_batch_size: int,
                         eos_token_id: int) -> Generator[PPOSample, None, None]:
    """
    用reference model对样本进行评分
    """

    for batch in chunk_n(scored_sample, training_model_eval_batch_size):
        query_ids = left_pad_sequence([sample.sample.query_ids for sample in batch], padding_value=eos_token_id)
        response_ids = pad_sequence([sample.sample.response_ids for sample in batch], padding_value=eos_token_id,
                                    batch_first=True)
        max_query_len = query_ids.shape[1]
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = input_ids.not_equal(eos_token_id).long()
        device = training_model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            training_model.eval()
            output = training_model(input_ids=input_ids, attention_mask=attention_mask)
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
    """
    调整reward
    """
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
        training_model: Callable[..., ActorCriticOutput],
        max_query_length: int,
        max_response_length: int,
        adjust_reward: Optional[Callable[[List[PPOSample]], None]] = None,
        generate_batch_size: int = 8,
        generate_kwargs: Optional[Dict] = None,
        scorer_batch_size: int = 8,
        training_model_eval_batch_size: int = 8,
        adjust_reward_batch_size: int = 8,
) -> SampleStore[PPOSample]:
    """
    生成PPO samples for training

    :param query: query iterator
    :param tokenizer: 分词器
    :param scorer: 评价函数
    :param generation_model: 生成模型
    :param training_model: 参考模型, 用于计算参考模型的 prob, value
    :param max_query_length: query最大长度
    :param max_response_length: response最大长度
    :param adjust_reward: 调整reward的函数, 用于增加 kl散度约束。可为空
    :param generate_batch_size:  生成过程的batch size
    :param generate_kwargs: 生成过程的参数
    :param scorer_batch_size: 评价过程的batch size
    :param training_model_eval_batch_size: 参考模型的batch size
    :param adjust_reward_batch_size: 调整reward的batch size
    :return: 生成的PPO样本
    """
    # 设置默认生成配置
    if generate_kwargs is None:
        generate_kwargs = dict()
    if 'num_beams' not in generate_kwargs:
        generate_kwargs['num_beams'] = 4
    if 'num_return_sequences' not in generate_kwargs:
        generate_kwargs['num_return_sequences'] = 4
    if 'do_sample' not in generate_kwargs:
        generate_kwargs['do_sample'] = False
    if 'early_stopping' not in generate_kwargs:
        generate_kwargs['early_stopping'] = True

    store = SampleStore[PPOSample]()

    # 生成PPOSample的过程主要分为4步
    # 1. 使用 generation model生成 response 文本
    # 2. 使用 scorer评价 query + response 文本
    # 3. 调用 reference model计算 prob, value
    # 4. 调整 reward， 例如增加KL散度约束等
    #
    # NOTE: 这四个步骤目前都可以设置独立的batch size，故使用generator做中间过程
    # NOTE: 每一步的输出尽量干净。即使某一步内部计算的时候需要padding，也应该在输出的时候去掉padding
    # NOTE: 因为可能生成很多PPOSample，所以中间结果都放到CPU中
    # NOTE: 如果速度不够，可以考虑多线程的生产者消费者模式

    generated_samples = _generate_response_by_generation_model(
        query=query, tokenizer=tokenizer, generation_model=generation_model,
        max_response_length=max_response_length, max_query_length=max_query_length,
        generate_batch_size=generate_batch_size, generate_kwargs=generate_kwargs)
    scored_samples = _score_generated_sample(generated_samples, scorer, scorer_batch_size)
    ppo_samples = _call_training_model(scored_samples, training_model, training_model_eval_batch_size,
                                       tokenizer.eos_token_id)
    for ppo_sample in _adjust_reward(ppo_samples, adjust_reward, adjust_reward_batch_size):
        store.append(ppo_sample)
    return store
