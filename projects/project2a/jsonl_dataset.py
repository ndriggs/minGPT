from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import json

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, head=True):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                self.data.append(json.loads(line)['text'])
                if (i > 10000) and head :
                    break
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = '<pad>'
        self.tokenizer.padding_side = 'left'

        # add ul2 sentinel tokens to tokenizer
        new_tokens = ['[S2S] ', '[NLU] ', '[NLG] '] + [f'<extra_id_{i}>' for i in range(100)]
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys()) 
        self.tokenizer.add_tokens(list(new_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx], padding='max_length', truncation=True, return_tensors="pt").input_ids[0]
        return tokens[:-1], tokens[1:]

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_block_size(self):
        return self.tokenizer.model_max_length
