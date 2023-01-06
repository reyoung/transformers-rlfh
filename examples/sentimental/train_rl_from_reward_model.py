from transformers_rlfh.scorers.best_of_n_scorers import BestOfNScorer


def main():
    scorer = BestOfNScorer.load_model('EleutherAI/gpt-neo-125M',
                                      "/data/tmp2/trial_57/epoch-29/pytorch_model.bin", "<|end of req rsp|>",
                                      device='cuda:0',
                                      last_token=False)
    print(scorer(["hello world"], ["hello world"])[0])


if __name__ == '__main__':
    main()
