
# For testing
import sys
from torch.utils.data import DataLoader
sys.path.append(sys.path[0] + '/../../')

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted

from cognitive import info_generator as ig
from cognitive import stim_generator as sg
from cognitive import task_bank as tb
from cognitive import constants as const

class StaticTaskDataset(Dataset):
    def __init__(self, root_dir):

        

        self.root_dir = root_dir
        # preprocessing steps for pretrained ResNet models
        self.transform = transforms.Compose([
                            transforms.Resize(224),
                            # transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

        # check the size of the dataset
        self.dataset_size = 0
        items = os.listdir(self.root_dir)
        for item in items:
            item_path = os.path.join(self.root_dir, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                self.dataset_size += 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        trial_path = os.path.join(self.root_dir, "trial%d"%idx)
        images = []
        
        for fp in natsorted(os.listdir(trial_path)):
            fp = os.path.join(trial_path, fp)

            if fp[-4:] == '.png':
                img = Image.open(fp)
                img = self.transform(img)
                images.append(img)
            elif 'trial_info' in fp:
                info = json.load(open(fp))
                
                actions = self._action_map(info["answers"])

                # npads = MAX_FRAMES - len(actions)
                # actions.extend([-1 for _ in range(0,npads)])
                
                instructions = info['instruction']
        
        images = torch.stack(images)

        return images, instructions, torch.tensor(actions), 
    
    def _action_map(self, actions):
        updated_actions = []
        for action in actions:
            if action == "null":
                updated_actions.append(2)
            elif action == "false":
                updated_actions.append(0)
            elif action == "true":
                updated_actions.append(1)
        return updated_actions


class DynamicTaskDataset(Dataset):
    def __init__(self, task_name, stim_dir, max_len, set_len, img_size=224, fixation_cue=True, train=True, task_path=None):
        self.stim_dir = stim_dir
        const.DATA = const.Data(dir_path=self.stim_dir)

        if task_path is None:
            self.task = tb.task_family_dict[task_name](whens=['last' + str(max_len-1), 'last0'])
        else:
            # CODE TO READ IN TASK from JSON
            raise Exception('not implemented whoops' )
        self.img_size = img_size

        self.fixation_cue = fixation_cue

        self.train = train

        self.frame_info = ig.FrameInfo(self.task, self.task.generate_objset())
        self.compo_info = ig.TaskInfoCompo(self.task, self.frame_info)

        self.transform = transforms.Compose([
                    transforms.Resize(224),
                    # transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        self.set_len = set_len

    def __len__(self):
        return self.set_len

    def __getitem__(self, idx):
        const.DATA = const.Data(dir_path=self.stim_dir)

        imgs, instructions, action = self.compo_info.generate_trial(train=self.train)
        for i, img in enumerate(imgs):
            imgs[i] = self.transform(img)
        actions = self._action_map(action)
        imgs = torch.stack(imgs)

        return imgs, instructions, torch.tensor(actions)
    
    def _action_map(self, actions):
        updated_actions = []
        for action in actions:
            if action == "null":
                updated_actions.append(2)
            elif action == "false":
                updated_actions.append(0)
            elif action == "true":
                updated_actions.append(1)
        return updated_actions