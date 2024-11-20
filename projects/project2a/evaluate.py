from mingpt.model import GPT
from transformers import AutoTokenizer
import torch
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=int)
    return parser.parse_args()

def evaluate_mingpt_hellaswag(checkpoint_path):
    from lm_eval import evaluator # strange workaround for a weird circular imports error
    from lm_eval.models.huggingface import HFLM 
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cuda'))
    lm = GPT(checkpoint['model_config'])
    lm.load_state_dict(checkpoint['model_state_dict'])
    lm = lm.to('cuda')
    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=lm, tokenizer=create_tokenizer(), device="cuda"),
        tasks=["hellaswag", "anli"], 
        num_fewshot=0,
        batch_size=32,
        limit=250,
    )
    return results # results['results']['hellaswag']['acc,none'], 

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

def main() :
    results = evaluate_mingpt_hellaswag(f'projects/project2a/xl_checkpoint_{sys.argv[1]}.pth') 
    print('anli_r1', results['results']['anli_r1']['acc,none'])
    print('anli_r2', results['results']['anli_r2']['acc,none'])
    print('anli_r3', results['results']['anli_r3']['acc,none'])
    print('hellaswag', results['results']['hellaswag']['acc,none'])

if __name__ == '__main__' :
    main()