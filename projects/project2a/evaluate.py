from mingpt.model import GPT
from transformers import AutoTokenizer
import torch

def evaluate_mingpt_hellaswag(checkpoint_path):
    from lm_eval import evaluator # strange workaround for a weird circular imports error
    from lm_eval.models.huggingface import HFLM 
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
    lm = GPT(checkpoint['model_config'])
    lm.load_state_dict(checkpoint['model_state_dict'])
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=lm, tokenizer=create_tokenizer(), device="cpu"),
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=32,
        limit=10,
    )
    return results['results']['hellaswag']['acc,none']

def create_tokenizer() :
    # recreate the same tokenizer used in training
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = '<pad>'
    tokenizer.padding_side = 'left'

    # add ul2 sentinel tokens to tokenizer
    new_tokens = ['[S2S] ', '[NLU] ', '[NLG] '] + [f'<extra_id_{i}>' for i in range(100)]
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys()) 
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer 

if __name__ == '__main__' :
    score = evaluate_mingpt_hellaswag("checkpoint_25200.pth")
    print(score)