from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import json
import numpy as np

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, head=False):
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
        while len(self.data[idx]) <= 20 :
            idx = (idx + 1) % len(self.data)
        # normal training
        # tokens = self.tokenizer(self.data[idx], truncation=True, return_tensors="pt").input_ids[0]
        # return tokens[:-1].long(), tokens[1:].long()
        # finetuning
        denoising_type = np.random.choice(['S', 'R', 'X'], p=[0.5, 0.25, 0.25])
        if denoising_type == 'S' :
            tokens, targets_start = self.sequential_denoising(self.data[idx])
        elif denoising_type == 'R' :
            tokens, targets_start = self.regular_denoising(self.data[idx])
        elif denoising_type == 'X' :            
            tokens, targets_start = self.extreme_denoising(self.data[idx])
        targets = tokens[1:].clone()    
        targets[:targets_start-1] = torch.full((targets_start-1,), -1, dtype=torch.int64)           
        return targets_start, tokens[:-1].long(), targets.long()

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_block_size(self):
        return self.tokenizer.model_max_length
    
    
    def generic_denoising(self, text:str, percent_tokens_corrupted: float, 
                          min_span_len:int, max_span_len: int, denoising_type: str) :
        average_span_length = (max_span_len + min_span_len) / 2
        max_tokens = np.round(1012 / (1 + ((2*percent_tokens_corrupted)/average_span_length))).astype(int) # this took some math
        tokens = self.tokenizer(text, truncation=True, max_length=max_tokens).input_ids
        num_corrupted_spans = np.round((len(tokens)*percent_tokens_corrupted)/average_span_length).astype(int)
        if num_corrupted_spans == 0 :
            return torch.LongTensor(tokens), 1
        total_gap = 0
        while total_gap <= num_corrupted_spans + 1 :
            span_lengths = np.random.randint(min_span_len, max_span_len+1, size=num_corrupted_spans)
            total_gap = len(tokens) - np.sum(span_lengths)
        gaps = np.random.choice(np.arange(1, total_gap), size=num_corrupted_spans, replace=False)
        gaps.sort()
        span_starts = np.array([gaps[i]+np.sum(span_lengths[:i]) for i in range(len(span_lengths))])
        i = 0
        while any([any((span_start > span_starts) & (span_start <= span_starts + span_lengths)) for span_start in span_starts]) \
             or any(span_starts + span_lengths > len(tokens)) :
            i += 1
            if i >= 50 :
                # we couldn't get a span set that worked, so don't corrupt, maybe not enough tokens
                # avoids getting stuck forever in a loop
                return torch.LongTensor(tokens), 1 
            span_lengths = np.random.randint(min_span_len, max_span_len+1, size=num_corrupted_spans)
            total_gap = len(tokens) - np.sum(span_lengths)
            gaps = np.random.choice(np.arange(1, total_gap), size=num_corrupted_spans, replace=False)
            gaps.sort()
            span_starts = np.array([gaps[i]+np.sum(span_lengths[:i]) for i in range(len(span_lengths))])
        if denoising_type == 'R' :
            new_input_tokens = [self.tokenizer('[NLU] ').input_ids[0]]
        elif denoising_type == 'X' :
            new_input_tokens = [self.tokenizer('[NLG] ').input_ids[0]]
        targets = []
        last_span_start = 0
        last_span_length = 0
        for i, (span_length, span_start) in enumerate(zip(span_lengths, span_starts)) :
            new_input_tokens.extend(list(tokens[last_span_start+last_span_length:span_start]))
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
    
    def regular_denoising(self, text) :
        if np.random.randint(2) == 1 :
            return self.generic_denoising(text, percent_tokens_corrupted=0.15, 
                                          min_span_len=2, max_span_len=5, denoising_type='R')
        else : 
            return self.generic_denoising(text, percent_tokens_corrupted=0.15,
                                          min_span_len=6, max_span_len=10, denoising_type='R')
    
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
        # I used the parameters from Table 1 in the UL2 paper
        rand_int = np.random.randint(4)
        if rand_int == 0 :
            return self.generic_denoising(text, percent_tokens_corrupted=0.5, 
                                          min_span_len=2, max_span_len=5,
                                          denoising_type='X')
        elif rand_int == 1 :
            return self.generic_denoising(text, percent_tokens_corrupted=0.5, 
                                          min_span_len=6, max_span_len=10,
                                          denoising_type='X')
        elif rand_int == 2 :
            return self.generic_denoising(text, percent_tokens_corrupted=0.15, 
                                          min_span_len=58, max_span_len=70,
                                          denoising_type='X')
        elif rand_int == 3 :
            return self.generic_denoising(text, percent_tokens_corrupted=0.5, 
                                          min_span_len=58, max_span_len=70,
                                          denoising_type='X')
        


