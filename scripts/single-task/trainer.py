import os
import torch
from torch.utils.data import DataLoader
from dsets import DynamicTaskDataset
from datetime import datetime
import json

class Trainer(object):
    def __init__(self, train_data, val_data, device, out_dir, args, static=True):
        assert type(train_data) == DataLoader or type(train_data) == DynamicTaskDataset, "train_data should be a torch DataLoader or class DynamicTaskDataset"
        assert type(val_data) == DataLoader or type(val_data) == DynamicTaskDataset, "val_data should be a torch DataLoader or class DynamicTaskDataset"

        self.device = device

        self.train_set = train_data
        self.val_set = val_data
        self.static = static

        self.all_loss = {'train_null_loss':[],'train_non_null_loss':[], 'val_null_loss':[], 'val_non_null_loss':[]}
        self.all_acc = {'train_null_acc':[], 'train_non_null_acc':[], 'val_null_acc':[], 'val_non_null_acc':[]}


        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir
        self.args = args

    def train(self, model, ins_encoder, criterion, optimizer, scheduler=None, epochs=100, batch_size=256):
        for epoch in range(epochs):
            print('epoch: ', epoch)
            i = 0

            null_accs = []
            non_null_accs = []
            null_losses = []
            non_null_losses = []
            for images, instructions, actions in self.train_set:
                model.train()
                optimizer.zero_grad()

                instructions = ins_encoder(instructions)

                output = model(images.to(self.device), instructions) # (seq_len, batch, n_classes)

                null_loss, non_null_loss, scale = self.loss(criterion, output.permute(1,0,2).reshape(-1,3), actions.type(torch.LongTensor).reshape(-1).to(self.device))
                null_losses.append(null_loss.item())
                non_null_losses.append(non_null_loss.item())

                if i%(epochs//2) != 0:
                    null_loss = 0
                null_loss = 0
                train_loss = null_loss + non_null_loss # *scale *(1/(scale**(epoch/2)))
                train_loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 2)
                predicted = predicted.permute(1,0).reshape(-1)

                null_acc, non_null_acc = self.correct(predicted, actions.reshape(-1).to(self.device))
                null_accs.append(null_acc.item())
                non_null_accs.append(non_null_acc.item())
                
                if scheduler is not None:
                    scheduler.step(epoch + i / len(self.train_set))

                i += 1
            self.stat_track('train', null_accs, non_null_accs, null_losses, non_null_losses)
            self.print_acc('train', null_accs, non_null_accs)
            if epoch%((epochs+1)//4) == 0 or epoch == epochs-1:
                self.validate(model, criterion, ins_encoder, batch_size)
                self.write_stats()

    def validate(self, model, criterion, ins_encoder, batch_size):
        null_accs = []
        non_null_accs = []
        null_losses = []
        non_null_losses = []

        for images, instructions, actions in self.val_set:
            model.eval()
            output = model(images.to(self.device), instructions)

            null_loss, non_null_loss, scale = self.loss(criterion, output.permute(1,0,2).reshape(-1,3), actions.type(torch.LongTensor).reshape(-1).to(self.device))
            null_losses.append(null_loss.item())
            non_null_losses.append(non_null_loss.item())
            
            _, predicted = torch.max(output.data, 2)
            predicted = predicted.permute(1,0).reshape(-1)

            null_acc, non_null_acc = self.correct(predicted, actions.reshape(-1).to(self.device))
            null_accs.append(null_acc.item())
            non_null_accs.append(non_null_acc.item())

        self.stat_track('val', null_accs, non_null_accs, null_losses, non_null_losses)
        self.print_acc('val', null_accs, non_null_accs)


    # Calculates the number of correct null action predictions and the number of correct non-null action predictions
    def correct(self, preds, targs):
        null_idxs = torch.where(targs.cpu() == 2)
        non_null_idxs = torch.where(targs.cpu() < 2)
        
        null_preds = preds[null_idxs]
        non_null_preds = preds[non_null_idxs]
        
        c_null = torch.sum(null_preds == targs[null_idxs])
        n_null = len(null_preds)
        null_acc = c_null/n_null
        
        c_non_null = torch.sum(non_null_preds == targs[non_null_idxs])
        n_non_null = len(non_null_preds)
        non_null_acc = c_non_null/n_non_null
        
        return null_acc, non_null_acc

    # Calculates the loss for a forward pass for both null and non-null action predictions (this is to avoid overfitting to null actions)
    def loss(self, criterion, preds, targs):
        # Find indexes of null frames and non-null frames
        null_idxs = torch.where(targs.cpu() == 2)
        non_null_idxs = torch.where(targs.cpu() < 2)
        
        # Add batch dimension and reorder into (batch_size, n_classes, seq_len)
        null_preds = preds[null_idxs]
        non_null_preds = preds[non_null_idxs]

        null_loss = criterion(null_preds, targs[null_idxs])
        non_null_loss = criterion(non_null_preds, targs[non_null_idxs])
        
        return null_loss, non_null_loss, len(null_idxs)/len(non_null_idxs)

    def stat_track(self, mode, null_accs, non_null_accs, null_losses, non_null_losses):
        assert mode in ['train', 'val'], 'mode must be either \"train\" or \"val\"'
        self.all_loss[mode + '_null_loss'].append((sum(null_losses)/len(null_losses)))
        self.all_loss[mode + '_non_null_loss'].append((sum(non_null_losses)/len(non_null_losses)))
        self.all_acc[mode + '_null_acc'].append((sum(null_accs)/len(null_accs)))
        self.all_acc[mode + '_non_null_acc'].append((sum(non_null_accs)/len(non_null_accs)))

    def print_acc(self, mode, null_accs, non_null_accs):
        assert mode in ['train', 'val'], 'mode must be either \"train\" or \"val\"'
        print(mode + ' null acc: ', (sum(null_accs)/len(null_accs)))
        print(mode + ' non-null acc: ', (sum(non_null_accs)/len(non_null_accs)))

    def write_stats(self):
        now = datetime.now().strftime("%d:%H:%M:%S")
        with open(self.out_dir  + 'log_' + now + '.json', 'w') as outfile: 
            log = self.args.copy()
            log.update(self.all_acc)
            log.update(self.all_loss)
            json.dump(log, outfile)

    def get_stats(self):
        return self.all_loss, self.all_acc

