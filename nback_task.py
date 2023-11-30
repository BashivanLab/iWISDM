"""Script for generating a COG dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import timeit
import os
import re
import random
import traceback
from collections import defaultdict
from typing import List, Dict
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import cv2
import gzip
import itertools
import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
from cognitive import task_generator as tg
import cognitive.task_bank as task_bank
import cognitive.constants as const
import cognitive.info_generator as ig
from cognitive.arguments import get_args

from main import generate_temporal_example

class NBACK_TaskDataset(Dataset):
    def __init__(self, stim_dir, families = ["CompareLoc"], whens = ['last1', 'last0', ], 
                seq_len = 6, nback = 2, 
                phase = "train", dataset_size = 2560, train = True):
        self.phase = phase
        self.stim_dir = stim_dir
        self.dataset_size = dataset_size
        self.families = families
        self.whens = whens
        const.DATA = const.Data(self.stim_dir)
        self.train = train
        self.composition = seq_len - nback
        
        families_count = defaultdict(lambda: 0)
        # you can also composite multiple DMS of different features together
        if len(families) == 1:
            self.families = families * self.composition
        assert len(self.families) == self.composition

        self.first_shareable = 1
        # preprocessing steps for pretrained ResNet models
        self.transform = transforms.Compose([
                            transforms.Resize(224),
                            # transforms.CenterCrop(224), # todo: to delete for shapenet task; why?
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
        self.reset()
    def reset(self):
        self.task_family = np.random.permutation(self.families)
        self.compo_tasks = [generate_temporal_example(task_family=[family], 
                            max_memory = 6, whens = self.whens, 
                            first_shareable = self.first_shareable) for family in self.task_family]


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        self.reset()
        const.DATA = const.Data(self.stim_dir, train = self.train)

        # temporal combination
        info = self.compo_tasks[0]
        for task in self.compo_tasks[1:]:
            info.merge(task)
        # info.temporal_switch() ### XLEI: why do we need temporal switch? I don't think that is the case

        imgs, ins, action = info.generate_trial(fixation_cue = False)
        
        for i, img in enumerate(imgs):
            imgs[i] = self.transform(img)
        actions = self._action_map(action)
        
        imgs = torch.stack(imgs)
        return imgs, "Instruction", torch.tensor(actions), 
    
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


# be sure whens and nback n match
# dst = NBACK_TaskDataset(stim_dir = "/mnt/store1/shared/XLshared_large_files/new_shapenet_train", 
#                 families = ["CompareLoc"], whens = ['last2', 'last0', ], 
#                 seq_len = 6, nback = 2, 
#                 phase = "train", dataset_size = 10)

# for j in range(4):
#     imgs, ins, actions = dst[j]
#     print(actions)
#     for i in range(len(imgs)):
#         plt.figure()
#         plt.imshow(imgs[i].permute(1,2,0))
#         plt.title("action: %d"%actions[i])
#         plt.savefig("/mnt/store1/xiaoxuan/sanity_check/trial%dframe%d.png"%(j,i))




# make sure all the tasks to be combined are DMS tasks
# assert all('Compare' in family for family in args.families)
# # the number of frames for each task is determined by n_back
# # e.g. in 2_back tasks, individual tasks have 3 frames, we compose tasks based on the total length of the task
# whens = [f'last{args.nback}', 'last0']
# composition = args.nback_length - args.nback  # the number of compositions
# print("number of composition:", composition)
# assert len(args.families) == composition
# first_shareable = 1

# #  e.g. families = [CompareLoc, CompareCat] could have 4 different task orders
# families_count = defaultdict(lambda: 0)
# families = args.families

# task_family = np.random.permutation(families)
# compo_tasks = [generate_temporal_example(task_family=[family], max_memory = 3, whens = whens, first_shareable = first_shareable) for family in task_family]
            
# # temporal combination
# info = compo_tasks[0]
# for task in compo_tasks[1:]:
#     info.merge(task)
# # info.temporal_switch() ### XLEI: why do we need temporal switch? I don't think that is the case

# imgs, ins, actions = info.generate_trial(fixation_cue = False)

# print(len(imgs))
# print(np.array(imgs[0]).shape)

# for i in range(len(imgs)):
#     plt.figure()
#     plt.imshow(np.array(imgs[i]))
#     plt.title("action: %s"%actions[i])
#     plt.savefig("/mnt/store1/xiaoxuan/sanity_check/trial%dframe%d.png"%(0,i))
