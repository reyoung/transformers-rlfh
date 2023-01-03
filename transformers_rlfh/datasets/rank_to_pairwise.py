from datasets import DatasetDict

__all__ = ['convert_rank_dataset_to_pairwise']


def mapper(batch):
    query = batch["query"]
    samples = zip(*[batch[f"sample{i}"] for i in range(4)])
    best = batch["best"]

    return_query = [None] * (len(query) * 3)
    return_good_sample = [None] * (len(query) * 3)
    return_bad_sample = [None] * (len(query) * 3)

    offset = 0

    for _query, _best, _samples in zip(query, best, samples):
        for i, sample in enumerate(_samples):
            if i == best:
                continue

            return_query[offset] = _query
            return_good_sample[offset] = _samples[_best]
            return_bad_sample[offset] = sample

    return {
        "query": return_query,
        "good_sample": return_good_sample,
        "bad_sample": return_bad_sample,
    }


def convert_rank_dataset_to_pairwise(ds: DatasetDict) -> DatasetDict:
    """
    将RLFH rank的数据集转换成pairwise数据集
    输入columns
        * query
        * sample0
        * sample1
        * sample2
        * sample3
        * best: int
    输出:
        query, good_sample, bad_sample

    每一个输入，可以产生三个输出。即 best 的sample，比其他的sample要更好
    """
    return ds.map(mapper, batched=True, remove_columns=["sample0", "sample1", "sample2", "sample3", "best", "query"])
