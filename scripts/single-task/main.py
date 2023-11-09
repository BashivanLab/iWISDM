import sys
# sys.path.append('../../')
sys.path.append('/home/lucasg/mlti-dl')

# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'



import argparse

from cognitive import task_bank as tb

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torchvision
from torchvision import transforms

from dsets import *
from models import *
from trainer import Trainer
from scripts.plot_dict import plot_dict

parser = argparse.ArgumentParser()# Add an argument

parser.add_argument('--static', type=bool, default=True)# Parse the argument
parser.add_argument('--train_path', type=str, default='./datasets/train_big')# Parse the argument
parser.add_argument('--val_path', type=str, default='./datasets/val_big')# Parse the argument
parser.add_argument('--task_name', type=str, default='CompareCategoryTemporal')
parser.add_argument('--task_path', type=str, required=False)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--task_max_len', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--niters', type=int, required=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--imgm_path', type=str, default='./tutorials/offline_models/resnet/resnet')
parser.add_argument('--insm_path', type=str, default='./tutorials/offline_models/all-mpnet-base-v2')



def get_device():
    # Check if a GPU is available
    if torch.cuda.is_available():
        # Request GPU device 0
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        # If no GPU is available, fall back to CPU
        device = torch.device("cpu")
        print("No GPU available, using CPU.")
    
    return device

if __name__ == '__main__':

    args = parser.parse_args()
    assert not (args.task_name == None and args.task_path == None), 'task_name or task_path is required'

    device = get_device()

    img_encoder = torch.load(args.imgm_path, map_location=device)
    model = RNN(hidden_size=args.hidden_size, img_encoder=img_encoder, device=device, max_frames=args.task_max_len).to(device)

    ins_model = AutoModel.from_pretrained(args.insm_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.insm_path)

    ins_encoder = InsEncoder(ins_model, tokenizer, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, T_mult=1 )

    if args.static:
        train_set = DataLoader(StaticTaskDataset(args.train_path), batch_size=args.batch_size, shuffle=True)
        val_set = DataLoader(StaticTaskDataset(args.val_path), batch_size=args.batch_size, shuffle=False)
    else: 
        if args.task_path is None:

            task = tb.task_family_dict[args.task_name](whens=['last' + str(args.task_max_len), 'last0'])
        else:
            # CODE TO READ IN TASK from JSON
            print('not implemented whoops')
        train_set = DynamicTaskDataset(task, img_size=args.img_size, fixation_cue=True, train=True)
        val_set = DynamicTaskDataset(task, img_size=args.img_size, fixation_cue=True, train=False)

    trainer = Trainer(train_set, val_set, device, static=args.static)

    trainer.train(model, ins_encoder, criterion, optimizer, scheduler=scheduler, epochs=args.epochs, iterations=args.niters, batch_size=args.batch_size)
    all_loss, all_acc = trainer.get_stats()

    plot_dict(all_loss)
    plot_dict(all_acc)









