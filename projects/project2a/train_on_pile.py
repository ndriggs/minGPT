from torch.utils.data.dataloader import DataLoader
import torch

from mingpt.model import GPT
from mingpt.trainer import Trainer
from jsonl_dataset import JSONLDataset


dataset = JSONLDataset('/nobackup/archive/usr/dw87/pile_data_10.jsonl', head=True)

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = dataset.get_vocab_size()
model_config.block_size = dataset.get_block_size() # gpt2 max length
model = GPT(model_config)

train_config = Trainer.get_default_config()
train_config.learning_rate = 2e-6 
train_config.max_iters = 10000000
train_config.num_workers = 0
train_config.batch_size = 1
trainer = Trainer(train_config, model, dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

print('running trainer')
trainer.run()
