from mingpt.model import GPT
from mingpt.trainer import Trainer
from projects.project2a.jsonl_dataset import JSONLDataset
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_iter', type=int)
    return parser.parse_args()


def main() : 
    args = parse_args()

    dataset = JSONLDataset('/nobackup/archive/usr/dw87/pile_data_10.jsonl', head=False)
    checkpoint = torch.load(f'projects/project2a/xl_checkpoint_{args.starting_iter}.pth', weights_only=False)
    model_config = checkpoint['model_config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 2e-6 
    train_config.max_iters = 13500
    train_config.num_workers = 0
    train_config.batch_size = 1
    train_config.starting_iter = args.starting_iter
    trainer = Trainer(train_config, model, dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)

    trainer.run()

if __name__ == '__main__':
    main()