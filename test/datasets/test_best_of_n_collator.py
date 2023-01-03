from transformers_rlfh.datasets.best_of_n_collator import BestOfNCollator
from transformers import AutoTokenizer


def test_best_of_n_collator_without_padding():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_tokens(["<|end of req rsp|>"])
    special_token_id = len(tokenizer) - 1
    collator = BestOfNCollator(tokenizer, special_token="<|end of req rsp|>", n_best=2)
    result = collator(
        batch=[
            {
                "query": "Hello",
                "sample0": "world",
                "sample1": "friend",
                "best": 0,
            },
            {
                "query": "Hello again",
                "sample0": "world",
                "sample1": "friend",
                "best": 1,
            }
        ]
    )
    input_ids, best = result

    assert input_ids.shape == (2, 2, 5)
    assert best.shape == (2,)
    assert input_ids[0, 0, 0] == tokenizer.encode("Hello")[0]
    assert input_ids[0, 0, 1] == special_token_id
    assert input_ids[0, 0, 2] == tokenizer.encode("world")[0]
    assert input_ids[0, 0, 3] == special_token_id
    assert input_ids[0, 0, 4] == special_token_id
    assert input_ids[0, 1, 0] == tokenizer.encode("Hello")[0]
    assert input_ids[0, 1, 1] == special_token_id
    assert input_ids[0, 1, 2] == tokenizer.encode("friend")[0]
    assert input_ids[0, 1, 3] == special_token_id
    assert input_ids[0, 1, 4] == special_token_id
    assert input_ids[1, 0, 0] == tokenizer.encode("Hello again")[0]
    assert input_ids[1, 0, 1] == tokenizer.encode("Hello again")[1]
    assert input_ids[1, 0, 2] == special_token_id
    assert input_ids[1, 0, 3] == tokenizer.encode("world")[0]
    assert input_ids[1, 0, 4] == special_token_id
    assert input_ids[1, 1, 0] == tokenizer.encode("Hello again")[0]
    assert input_ids[1, 1, 1] == tokenizer.encode("Hello again")[1]
    assert input_ids[1, 1, 2] == special_token_id
    assert input_ids[1, 1, 3] == tokenizer.encode("friend")[0]
    assert input_ids[1, 1, 4] == special_token_id
    assert best[0] == 0
    assert best[1] == 1


def test_best_of_n_collator_with_padding():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.add_tokens(["<|end of req rsp|>"])
    special_token_id = len(tokenizer) - 1
    padding_token_id = tokenizer.eos_token_id
    collator = BestOfNCollator(tokenizer, special_token="<|end of req rsp|>", n_best=2,
                               pad_token_id=tokenizer.eos_token_id)
    result = collator(
        batch=[
            {
                "query": "Hello",
                "sample0": "world",
                "sample1": "friend",
                "best": 0,
            },
            {
                "query": "Hello again",
                "sample0": "world",
                "sample1": "friend",
                "best": 1,
            }
        ]
    )
    input_ids, best, pos = result
    assert input_ids.shape == (2, 2, 5)
    assert best.shape == (2,)
    assert pos.shape == (2, 2)
    assert input_ids[0, 0, 0] == tokenizer.encode("Hello")[0]
    assert input_ids[0, 0, 1] == special_token_id
    assert input_ids[0, 0, 2] == tokenizer.encode("world")[0]
    assert input_ids[0, 0, 3] == special_token_id
    assert input_ids[0, 0, 4] == padding_token_id
    assert input_ids[0, 1, 0] == tokenizer.encode("Hello")[0]
    assert input_ids[0, 1, 1] == special_token_id
    assert input_ids[0, 1, 2] == tokenizer.encode("friend")[0]
    assert input_ids[0, 1, 3] == special_token_id
    assert input_ids[0, 1, 4] == padding_token_id
    assert input_ids[1, 0, 0] == tokenizer.encode("Hello again")[0]
    assert input_ids[1, 0, 1] == tokenizer.encode("Hello again")[1]
    assert input_ids[1, 0, 2] == special_token_id
    assert input_ids[1, 0, 3] == tokenizer.encode("world")[0]
    assert input_ids[1, 0, 4] == special_token_id
    assert input_ids[1, 1, 0] == tokenizer.encode("Hello again")[0]
    assert input_ids[1, 1, 1] == tokenizer.encode("Hello again")[1]
    assert input_ids[1, 1, 2] == special_token_id
    assert input_ids[1, 1, 3] == tokenizer.encode("friend")[0]
    assert input_ids[1, 1, 4] == special_token_id
    assert best[0] == 0
    assert best[1] == 1
    assert pos[0, 0] == 3
    assert pos[0, 1] == 3
    assert pos[1, 0] == 4
    assert pos[1, 1] == 4
