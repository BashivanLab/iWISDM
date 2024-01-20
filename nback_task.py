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
from multiprocessing import Pool
import numpy as np
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
from cognitive import task_generator as tg
import cognitive.task_bank as task_bank
import cognitive.constants as const
import cognitive.info_generator as ig
from cognitive.arguments import get_args

from main import generate_temporal_example


class ParallelGen_NBACK_Task(Dataset):
    def __init__(
            self,
            stim_dir,
            families=["CompareLoc"],
            seq_len=6,
            nback=2,
            phase="train",
            train=True,
            output_dir=None,
            fixation_cue=True
    ):
        self.phase = phase
        self.train = train
        self.stim_dir = stim_dir
        const.DATA = const.Data(
            dir_path=self.stim_dir,
            train=self.train
        )

        self.families = families
        self.whens = [f'last{nback}', 'last0']
        self.composition = seq_len - nback
        self.output_dir = output_dir
        self.fixation_cue = fixation_cue
        self.img_size = 224

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
                                                      max_memory=6, whens=self.whens,
                                                      first_shareable=self.first_shareable) for family in
                            self.task_family]

    def generate_trial(self, idx):
        self.reset()
        const.DATA = const.Data(self.stim_dir, train=self.train)

        # temporal combination
        info = self.compo_tasks[0]
        for task in self.compo_tasks[1:]:
            info.merge(task)
            # info.temporal_switch() ### XLEI: why do we need temporal switch? I don't think that is the case
        # imgs, ins, action = info.generate_trial(fixation_cue = self.fixation_cue)

        # for i, img in enumerate(imgs):
        #     imgs[i] = self.transform(img)
        # actions = self._action_map(action)

        # imgs = torch.stack(imgs)

        fp = os.path.join(self.output_dir, 'trial' + str(idx))
        info.write_trial_instance(fp, self.img_size, self.fixation_cue)
        # self.write_trial_instance(fp, self.fixation_cue)
        # return imgs, ins, torch.tensor(actions)

    # Spawns the generation of n trials of the task into seperate cores 
    def generate_trials(self, n_trials):
        with Pool() as pool:
            pool.map(self.generate_trial, range(self.cur_idx, self.cur_idx + n_trials))
        self.cur_idx += n_trials

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


# # be sure whens and nback n match
# dst = ParallelGen_NBACK_Task(
#     stim_dir="/Users/markbai/Documents/COG_v3_shapenet/data/MULTIF_5_stim",
#     families=["CompareLoc"],
#     seq_len=6,
#     nback=1,
#     phase="train",
#     output_dir="/Users/markbai/Documents/COG_v3_shapenet/test",
# )
#
# dst.generate_trial(16)
# be sure whens and nback n match
# dst = ParallelGen_NBACK_Task(
#     stim_dir = "/mnt/store1/shared/XLshared_large_files/new_shapenet_train",
#     families = ["CompareLoc"],
#     seq_len = 6,
#     nback = 2,
#     phase = "train",
#     output_dir =  "/mnt/store1/xiaoxuan/sanity_check",
# )


# dst.generate_trial(16)



