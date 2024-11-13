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
        tokens = self.tokenizer(text, truncation=True, max_length=942).input_ids
        num_corrupted_spans = np.round((len(tokens)*percent_tokens_corrupted)/average_span_length).astype(int)
        span_starts = np.random.randint(len(tokens), size=num_corrupted_spans)
        span_starts.sort()
        span_lengths = np.random.randint(2,6, size=num_corrupted_spans)
        while any([any((span_start > span_starts) & (span_start <= span_starts + span_lengths)) for span_start in span_starts]) :
            span_starts = np.random.randint(len(tokens), size=num_corrupted_spans)
            span_starts.sort()
            span_lengths = np.random.randint(2,6, size=num_corrupted_spans)
        new_input_tokens = [self.tokenizer('[NLU] ').input_ids[0]]
        targets = []
        last_span_start = 0
        last_span_length = 0
        for i, (span_length, span_start) in enumerate(zip(span_lengths, span_starts)) :
            new_input_tokens.extend(list(tokens[last_span_start+last_span_length:span_starts]))
            new_input_tokens.append(self.tokenizer(f'<extra_id_{i}>').input_ids[0])
            targets.append(self.tokenizer(f'<extra_id_{i}>').input_ids[0])
            targets.extend(list(tokens[span_start:span_start+span_length]))
            last_span_start = span_start
            last_span_length = span_length
        new_input_tokens.extend(list(tokens[last_span_start+last_span_length:]))
        new_input_tokens.append(self.tokenizer.pad_token_id)
        # should combine and then add padding til its 1024
        sequence = torch.LongTensor(new_input_tokens + targets)
        targets_start = len(new_input_tokens)
        return sequence, targets_start
    
    def sequential_denoising(self, text) :
        tokens = self.tokenizer(text, truncation=True, max_length=1020).input_ids
        span_start = np.round(len(tokens)*0.75).astype(int)
        new_input_tokens = [self.tokenizer('[S2S] ').input_ids[0]] + list(tokens[:span_start]) \
            + [self.tokenizer('<extra_id_0>').input_ids[0], self.tokenizer.pad_token_id]
        targets = list(tokens[span_start:])
        sequence = torch.LongTensor(new_input_tokens + targets)
        targets_start = len(new_input_tokens)
        return sequence, targets_start
    
    def extreme_denoising(self, text) :
        pass


