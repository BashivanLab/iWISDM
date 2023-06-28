# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script for generating a COG dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import functools

import timeit
import cv2
from PIL import Image
import gzip
import itertools
import json
import multiprocessing
import os
import re
import random
import shutil
import traceback
from collections import defaultdict
import numpy as np
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
from cognitive import task_generator as tg
import cognitive.task_bank as task_bank
import cognitive.constants as const
import cognitive.info_generator as ig
from cognitive.arguments import get_args
import torch
from torch.utils.data import Dataset


try:
    range_fn = xrange  # py 2
except NameError:
    range_fn = range  # py 3

def action_map(action):
    if action == "false": return 0
    elif action =="true": return 1

def log_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            print('Exception in ' + func.__name__)
            traceback.print_exc()
            raise e

    return wrapped_func


class FileWriter(object):
    """Writes per_file examples in a file. Then, picks a new file."""

    def __init__(self, base_name, per_file=100, start_index=0, compress=True):
        self.per_file = per_file
        self.base_name = base_name
        self.compress = compress
        self.cur_file_index = start_index - 1
        self.cur_file = None
        self.written = 0
        self.file_names = []

        self._new_file()

    def _file_name(self):
        return '%s_%d.json' % (self.base_name, self.cur_file_index)

    def _new_file(self):
        if self.cur_file:
            self.close()

        self.written = 0
        self.cur_file_index += 1
        # 'b' is needed because we want to seek from the end. Text files
        # don't allow seeking from the end (because width of one char is
        # not fixed)
        self.cur_file = open(self._file_name(), 'wb')
        self.file_names.append(self._file_name())

    def write(self, data):
        if self.written >= self.per_file:
            self._new_file()
        self.cur_file.write(data)
        self.cur_file.write(b'\n')
        self.written += 1

    def close(self):
        self.cur_file.seek(-1, os.SEEK_END)
        # Remove last new line
        self.cur_file.truncate()
        self.cur_file.close()

        if self.compress:
            # Compress the file and delete the original. We can write to compressed
            # file immediately because truncate() does not work on compressed files.
            with open(self._file_name(), 'rb') as f_in, \
                    gzip.open(self._file_name() + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(self._file_name())



def output_task_instance(task_info, img_size, fixation_cue=False):
    objset = task_info.frame_info.objset
    imgs = []
    for i, (epoch, frame) in enumerate(zip(sg.render(objset, img_size), task_info.frame_info)):
        print(i)
        if fixation_cue:
            if not any('ending' in description for description in frame.description):
                sg.add_fixation_cue(epoch)
        # img = Image.fromarray(epoch, 'RGB')

        imgs.append(epoch)

    examples, compo_example, memory_info = task_info.get_examples()
    actions = [2] ##### xlei: need to be modified for other tasks, I only dealt with tasks with first frame no action
    ins = []
    curr_max_epochs = 0
    for i, task_example in enumerate(examples):
        curr_actions = task_example["answers"]
        curr_epochs = task_example["epochs"]
        actions.append(action_map(curr_actions[0]))
        if i == 0:
            ins.append(task_example["question"])


    return np.stack(imgs), actions, ins




# TODO: move to stim_generator
def write_task_instance(fname, task_info, img_size, fixation_cue=True):
    if not os.path.exists(fname):
        os.makedirs(fname)
    objset = task_info.frame_info.objset
    for i, (epoch, frame) in enumerate(zip(sg.render(objset, img_size), task_info.frame_info)):
        if fixation_cue:
            if not any('ending' in description for description in frame.description):
                sg.add_fixation_cue(epoch)
        img = Image.fromarray(epoch, 'RGB')
        filename = os.path.join(fname, f'epoch{i}.png')
        img.save(filename)

    examples, compo_example, memory_info = task_info.get_examples()
    for i, task_example in enumerate(examples):
        filename = os.path.join(fname, f'task{i} example')
        with open(filename, 'w') as f:
            json.dump(task_example, f, indent=4)

    filename = os.path.join(fname, 'compo_task_example')
    with open(filename, 'w') as f:
        json.dump(compo_example, f, indent=4)

    filename = os.path.join(fname, 'memory_trace_info')
    with open(filename, 'w') as f:
        json.dump(memory_info, f, indent=4)

    filename = os.path.join(fname, 'frame_info')
    with open(filename, 'w') as f:
        json.dump(task_info.frame_info.dump(), f, indent=4)


def generate_temporal_example(max_memory, max_distractors, task_family,
                              whens=None, first_shareable=None, *args, **kwargs):
    """
    generate 1 task objset and composition info object

    :param max_memory:
    :param max_distractors:
    :param task_family:
    :param whens:
    :param first_shareable:
    :return:
    """
    task = task_bank.random_task(task_family, whens, first_shareable)
    assert isinstance(task, tg.TemporalTask)

    # To get maximum memory duration, we need to specify the following average
    # memory value
    avg_mem = round(max_memory / 3.0 + 0.01, 2)
    if max_distractors == 0:
        objset = task.generate_objset(average_memory_span=avg_mem, *args, **kwargs)
    else:
        objset = task.generate_objset(n_distractor=random.randint(1, max_distractors),
                                      average_memory_span=avg_mem, *args, **kwargs)
    # Getting targets can remove some objects from objset.
    # Create example fields after this call.
    frame_info = ig.FrameInfo(task, objset)
    compo_info = ig.TaskInfoCompo(task, frame_info)
    return compo_info


def generate_compo_temporal_example(max_memory, max_distractors, families, n_tasks=1,
                                    *args, **kwargs):
    '''

    :param first_shareable:
    :param whens:
    :param families:
    :param max_memory:
    :param max_distractors:
    :param n_tasks:
    :return: combined TaskInfo Compo
    '''

    whens = kwargs.pop('whens', [None for _ in range(n_tasks)])

    if not isinstance(whens[0], list):
        whens = [whens for _ in range(n_tasks)]
    if n_tasks == 1:
        return generate_temporal_example(max_memory, max_distractors, families, whens=whens[0], *args, **kwargs)

    compo_tasks = [generate_temporal_example(max_memory, max_distractors, families, whens=whens[i], *args, **kwargs)
                   for i in range(n_tasks)]
    # temporal combination
    cur_task = compo_tasks[0]
    for task in compo_tasks[1:]:
        cur_task.merge(task)
    return cur_task


# TODO: split training and validation after task generation

class nn_generate_dataset:
    def __init__(self, random_families=True,
                     composition=1, img_size=224,
                     fixation_cue=True,):
        self.examples_per_family = 1
        self.random_families = random_families

        self.composition = composition
        self.img_size = img_size
        self.fixation_cue = fixation_cue

        if not self.random_families:
            assert self.families is not None
            assert self.composition == len(self.families)

        self.families_count = defaultdict(lambda: 0)

    def sample_gen(self, families=None, *args, **kwargs):
        if families is None or families == 'all':
            families = list(task_bank.task_family_dict.keys())
        n_families = len(families)
        p = np.random.permutation(n_families)

        if self.random_families:
            task_family = list()
            for j in range(self.composition):
                task_family.append(families[p[j] % n_families])
                self.families_count[families[p[j] % n_families]] += 1

                info = generate_compo_temporal_example(families=task_family, n_tasks=self.composition, *args, **kwargs)


                img, actions, ins = output_task_instance(info, self.img_size, self.fixation_cue)
                ins = ins * len(actions)

            # xlei: not sure about the composition
        return img, actions, ins



class multfsDataset(Dataset):
    def __init__(self, task_gen_func):
        self.task_gen_func = task_gen_func

    def __len__(self):
        return 1200

    # xlei: need to make further modification: how to avoid feeding in *args and **kwargs when get items?
    def reset(self, families, *args, **kwargs):
        self.img, self.actions, self.ins = self.task_gen_func.sample_gen(families=families, *args, **kwargs)

    def __getitem__(self, index):
        return self.img, self.actions, self.ins


if __name__ == '__main__':
    args = get_args()
    print(args)

    const.DATA = const.Data(args.stim_dir)

    whens = [f'last{args.nback}', 'last0']
    task_gen_func = nn_generate_dataset( random_families=args.random_families,
                 composition=1, img_size=224,
                 fixation_cue=True,)
    dt = multfsDataset(task_gen_func)
    dt.reset(families = args.families, max_memory=args.max_memory, max_distractors=args.max_distractors,
                                         whens=whens, first_shareable=1, temporal_switch=args.temporal_switch)
    img, actions, ins = dt[0]
    print("-------------")
    print(img.shape)
    print(len(actions))
    print(len(ins))
