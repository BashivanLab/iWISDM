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

import functools
import timeit
import os
import re
import random
import traceback
from collections import defaultdict
from typing import List, Dict

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

try:
    range_fn = xrange  # py 2
except NameError:
    range_fn = range  # py 3


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


def generate_temporal_example(max_memory: int, task_family: List[str],
                              whens: List[str] = None, first_shareable: int = None,
                              *args, **kwargs):
    """generate 1 task objset and composition info object

    :param max_memory: used to generate objset, determines maximum task
    :param task_family: list of task names to choose from
    :param whens: the last_i of the tasks, determines task length
    :param first_shareable: determines how tasks are combined
    :return:
    """
    task = task_bank.random_task(task_family, whens, first_shareable)
    assert isinstance(task, tg.TemporalTask)

    # To get maximum memory duration, we need to specify the following average
    # memory value
    avg_mem = round(max_memory / 3.0 + 0.01, 2)
    objset = task.generate_objset(average_memory_span=avg_mem, *args, **kwargs)
    # Getting targets can remove some objects from objset.
    # Create example fields after this call.
    frame_info = ig.FrameInfo(task, objset)
    compo_info = ig.TaskInfoCompo(task, frame_info)
    return compo_info


def generate_compo_temporal_example(max_memory: int,
                                    families: List[str],
                                    n_tasks: int = 1,
                                    whens: List[List[str]] = None,
                                    *args, **kwargs) -> ig.TaskInfoCompo:
    if whens is None:
        whens = [None for _ in range(n_tasks)]
    if not isinstance(whens[0], list):
        whens = [whens for _ in range(n_tasks)]  # repeat the whens for each individual task

    if n_tasks == 1:
        return generate_temporal_example(max_memory, families, whens=whens[0], *args, **kwargs)

    compo_tasks = [generate_temporal_example(max_memory, families, whens=whens[i], *args, **kwargs)
                   for i in range(n_tasks)]
    # temporal combination
    cur_task = compo_tasks[0]
    for task in compo_tasks[1:]:
        cur_task.merge(task)
    return cur_task


# TODO: split training and validation after task generation

def generate_dataset(
        examples_per_family: int = 10,
        output_dir: str = './data',
        random_families: bool = True,
        families: List[str] = None,
        composition: int = 1,
        img_size: int = 224,
        train: float = 0.7,
        validation: float = 0.3,
        fixation_cue: bool = True,
        *args, **kwargs
) -> Dict[str, int]:
    if not random_families:
        assert families is not None
        assert composition == len(families)
    assert train + validation == 1.0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO: add task_family order instance folders
    #  e.g. families = [CompareLoc, CompareCat] could have 4 different task orders
    families_count = defaultdict(lambda: 0)
    if families is None or families == 'all':
        families = list(task_bank.task_family_dict.keys())
    print(families)
    n_families = len(families)
    total_examples = n_families * examples_per_family
    train_examples = total_examples * train
    validation_examples = total_examples * validation

    # creating task folder names
    fam_str = '_'.join(families)
    if train != 0.7 and validation != 0.3:
        base_fname = os.path.join(output_dir,
                                  f'tasks_{fam_str}_{train}_{validation}')
    else:
        base_fname = os.path.join(output_dir,
                                  f'tasks_{fam_str}')

    train_fname = os.path.join(base_fname, 'train')
    validation_fname = os.path.join(base_fname, 'validation')

    if not os.path.exists(train_fname):
        os.makedirs(train_fname)
    if not os.path.exists(validation_fname):
        os.makedirs(validation_fname)

    # TODO: temporal_switch generation pipeline. User specify compo vs switch or randomly.
    #  If randomly, how to split compo and switch?
    if random_families:
        p = np.random.permutation(total_examples)
        i = 0
        while i < len(p):
            if i % 10000 == 0 and i > 0:
                print("Generated ", i, " examples")

            task_family = list()
            for j in range(composition):
                task_family.append(families[p[i] % n_families])  # add a random task to be in composition
                families_count[families[p[i] % n_families]] += 1

            info = generate_compo_temporal_example(families=task_family, n_tasks=composition, *args, **kwargs)
            # Write the example randomly to training or validation folder
            split = bool(random.getrandbits(1))
            if (split or validation_examples <= 0) and train_examples > 0:
                train_examples -= 1
                fname = os.path.join(train_fname, f'{i}')
            else:
                validation_examples -= 1
                fname = os.path.join(validation_fname, f'{i}')

            # xlei: pass train/val parameter to write trial_instances
            if i < int(total_examples * train):
                info.write_trial_instance(fname, img_size, fixation_cue, train = True)
            else:
                info.write_trial_instance(fname, img_size, fixation_cue, train=False)
            i += 1
    else:
        i = 0
        while i < total_examples:
            if i % 10000 == 0 and i > 0:
                print("Generated ", i, " examples")
            task_family = np.random.permutation(families)

            compo_tasks = [generate_temporal_example(task_family=[family], *args, **kwargs)
                           for family in task_family]
            # temporal combination
            info = compo_tasks[0]
            for task in compo_tasks[1:]:
                info.merge(task)
            # TODO: find boolean output task and apply temporal switch
            #  draw diagram for branching structure of temporal composition/switch
            #  put in info_generator?
            info.temporal_switch()

            # Write the example randomly to training or validation folder
            split = bool(random.getrandbits(1))
            if (split or validation_examples <= 0) and train_examples > 0:
                train_examples -= 1
                fname = os.path.join(train_fname, f'{i}')
            else:
                validation_examples -= 1
                fname = os.path.join(validation_fname, f'{i}')

            info.write_trial_instance(fname, img_size, fixation_cue)
            i += 1
    return families_count


# examples of how to generate composition task datasets
def main():
    args = get_args()
    print(args)

    const.DATA = const.Data(args.stim_dir)

    start = timeit.default_timer()
    # whens are lists of lists of strs that determine the lengths of each task
    # also determines which frame each stimuli is placed
    # e.g. [[last2, last0],  [last1, last0]]
    # the way tasks are composed are determined by this, and the first_shareable arg
    if args.nback > 0:
        assert all('Compare' in family for family in args.families)
        # the number of frames for each task is determined by n_back
        # e.g. in 2_back tasks, individual tasks have 3 frames, we compose tasks based on the total length of the task
        whens = [f'last{args.nback}', 'last0']
        composition = args.nback_length - args.nback + 1  # the number of compositions
        first_shareable = 1
        generate_dataset(examples_per_family=args.trials_per_family, output_dir=args.output_dir,
                         composition=composition, img_size=args.img_size,
                         random_families=args.non_random_families, families=args.families,
                         train=args.training, validation=args.validation, fixation_cue=args.fixation_cue,
                         max_memory=args.max_memory, whens=whens, first_shareable=first_shareable,
                         temporal_switch=args.temporal_switch)
    elif args.seq_length > 0:
        first_shareable = 1
        # in seq_reverse, the response frames of the new tasks appear before the old tasks
        if args.seq_reverse:
            # minimum 2 frames per DMS task
            if args.seq_length * 2 - 1 > const.MAX_MEMORY:
                raise ValueError('Total composition length too long, increase const.MAX_MEMORY')
            # e.g. [[last5, last0], [last3, last0], [last1, last0]]
            # then, the response frames of the new tasks would be shifted forward
            whens = [[f'last{const.MAX_MEMORY - (i * 2)}', 'last0'] for i in range(args.seq_length)]
        else:
            # this is basically interleave
            last_when = sg.random_when()
            while int(re.search(r'\d+', last_when).group()) - 1 < args.seq_length:
                last_when = sg.random_when()
            whens = [[last_when, 'last0'] for _ in range(args.seq_length)]
        generate_dataset(examples_per_family=args.trials_per_family, output_dir=args.output_dir,
                         composition=args.seq_length, img_size=args.img_size,
                         random_families=args.non_random_families, families=args.families,
                         train=args.training, validation=args.validation, fixation_cue=args.fixation_cue,
                         max_memory=args.max_memory, whens=whens, first_shareable=first_shareable,
                         temporal_switch=args.temporal_switch)
    else:
        whens = args.whens
        if args.fix_delay and whens is None:
            # TODO: move this into task.init
            whens = [f'last{const.DATA.MAX_MEMORY}', 'last0']
        generate_dataset(examples_per_family=args.trials_per_family, output_dir=args.output_dir,
                         composition=args.composition, img_size=args.img_size,
                         random_families=args.non_random_families, families=args.families,
                         train=args.training, validation=args.validation, fixation_cue=args.fixation_cue,
                         max_memory=args.max_memory, max_distractors=args.max_distractors,
                         whens=whens, first_shareable=args.first_shareable, temporal_switch=args.temporal_switch)

    # # from task_bank.task_family_dict
    # tasks = ['CompareLoc', 'CompareObject', 'CompareCategory', 'CompareViewAngle']
    # task_combs = dict()
    # # generate all possible task combinations with specified i composition
    #
    # for i in range(1, 4):
    #     task_combs[i] = list(itertools.combinations_with_replacement(tasks, i))
    #
    # for i, task_comb in task_combs.items(): # for each combination of tasks
    #     for task_fam in task_comb:
    #         if i == 1:
    #             generate_dataset(max_memory, max_distractors,
    #                          1000, '/Users/markbai/Documents/School/COMP402/COG_v3/data/all_stims/no_comp',
    #                          composition=i, families=task_fam, random_families=False)
    #         else:
    #             generate_dataset(max_memory, max_distractors,
    #                              1000, f'/Users/markbai/Documents/School/COMP402/COG_v3/data/all_stims/comp_{i}',
    #                              composition=i, families=task_fam, random_families=False)
    # generate_dataset(max_memory, max_distractors,
    #                  50000, '/Users/markbai/Documents/School/COMP402/COG_v3/data/test',
    #                  composition=1, families=['CompareCategory'], train=0.7, validation=0.3)

    stop = timeit.default_timer()

    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
