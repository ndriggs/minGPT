from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import json
import numpy as np

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
    
    def regular_denoising(self, text) :
        percent_tokens_corrupted = 0.15
        average_span_length = 3.5
        tokens = self.tokenizer(text, truncation=True, max_length=978).input_ids
        num_corrupted_spans = np.round((len(tokens)*percent_tokens_corrupted)/average_span_length).astype(int)
        span_positions = (len(tokens)/(num_corrupted_spans+1))*np.arange(num_corrupted_spans+1)
        new_input_tokens = [self.tokenizer('[NLU] ').input_ids[0]]
        targets = []
        span_length = 0
        for i, (span_position1, span_position2) in enumerate(zip(span_positions[:-1], span_positions[1:])) :
            new_input_tokens.extend(list(tokens[span_position1+span_length:span_position2]))
            new_input_tokens.append(self.tokenizer(f'<extra_id_{i}>').input_ids[0])
            targets.append(self.tokenizer(f'<extra_id_{i}>').input_ids[0])
            span_length = np.random.randint(2,6)
            targets.extend(list(tokens[span_position2:span_position2+span_length]))
        new_input_tokens.extend(list(tokens[span_positions[-1]+span_length:]))
        new_input_tokens.append(self.tokenizer.pad_token_id)
        sequence = torch.LongTensor(new_input_tokens + targets)
        targets_start = len(new_input_tokens)
        return sequence, targets_start

        