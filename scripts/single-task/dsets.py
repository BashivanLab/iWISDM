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
                print(fp)
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


class DynamicTaskDataset(object):
    def __init__(self, task, img_size=224, fixation_cue=True, train=True):
        self.task = task

        self.img_size = img_size

        self.fixation_cue = fixation_cue

        self.train = train

        self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __getitem__(self):
        frame_info = ig.FrameInfo(self.task, self.task.generate_objset())
        compo_info = ig.TaskInfoCompo(self.task, frame_info)
        objset = compo_info.frame_info.objset

        images = []
        for epoch, frame in zip(sg.render(objset, self.img_size, train=self.train), compo_info.frame_info):
            if self.fixation_cue:
                if not any('ending' in description for description in frame.description):
                    sg.add_fixation_cue(epoch)
            img = Image.fromarray(epoch, 'RGB')
            images.append(img)
        _, data, _ = compo_info.get_examples()

        images = np.stack(images)
        instruction = data['instructions']
        actions = self._action_map(data['answers'])

        return images, instruction, torch.tensor(actions)
    
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



# class DynamicTaskDataset(object):
#     def __init__(self, task, img_size=224, fixation_cue=True, train=True):
#         self.task = task

#         self.img_size = img_size

#         self.fixation_cue = fixation_cue

#         self.train = train

#         self.transform = transforms.Compose([
#                     transforms.Resize(224),
#                     transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ])

#     def get_sample(self):
#         frame_info = ig.FrameInfo(self.task, self.task.generate_objset())
#         compo_info = ig.TaskInfoCompo(self.task, frame_info)
#         objset = compo_info.frame_info.objset

#         images = []
#         for epoch, frame in zip(sg.render(objset, self.img_size, train=self.train), compo_info.frame_info):
#             if self.fixation_cue:
#                 if not any('ending' in description for description in frame.description):
#                     sg.add_fixation_cue(epoch)
#             img = Image.fromarray(epoch, 'RGB')
#             images.append(img)
#         _, data, _ = compo_info.get_examples()

#         images = np.stack(images)
#         instruction = data['instructions']
#         actions = self._action_map(data['answers'])

#         return images, instruction, torch.tensor(actions)
    
#     def get_samples(self, batch_size=128):
        
#         images_batch = torch.zeros()
#         ins_batch = 1
#         actions_batch = 1
#         for i in range(batch_size):
            
    
#     def _action_map(self, actions):
#         updated_actions = []
#         for action in actions:
#             if action == "null":
#                 updated_actions.append(2)
#             elif action == "false":
#                 updated_actions.append(0)
#             elif action == "true":
#                 updated_actions.append(1)
#         return updated_actions