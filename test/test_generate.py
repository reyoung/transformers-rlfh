import itertools

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_rlfh.generate import generate_ppo_samples
from transformers_rlfh.models.actor_critic import ActorCriticLM


def test_generate():
    generate_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    generate_model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    ac = ActorCriticLM(generate_model)

    store = generate_ppo_samples(
        ["EleutherAI has",
         "今天天气不错"],
        tokenizer, scorer=lambda q, r: [len(r_) for r_ in r], generation_model=generate_model,
        training_model=ac,
        max_query_length=100, max_response_length=32)
    print(len(store) != 0)
