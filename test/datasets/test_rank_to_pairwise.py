import os.path

import datasets
from transformers_rlfh.datasets.rank_to_pairwise import convert_rank_dataset_to_pairwise


def test_rank_to_pairwise():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sentimental", "data")
    ds = datasets.load_from_disk(path)
    prev_len = len(ds["train"])
    ds = convert_rank_dataset_to_pairwise(ds)
    new_len = len(ds["train"])
    assert new_len == 3 * prev_len
