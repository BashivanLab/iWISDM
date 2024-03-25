"""
Code adapted from 'A Dataset and Architecture for Visual Reasoning with a Working Memory', Guangyu Robert Yang, et al.
Paper: https://arxiv.org/abs/1803.06092
Code: https://github.com/google/cog
"""

"""Store all the constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from typing import Tuple
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2

# define all available operators for constructing graphs
LOGIC_OPS = ['And', 'Or', 'Xor', 'IsSame', "NotSame"]
BOOLEAN_OUT_OPS = ['IsSame', 'Exist', 'And', 'Or', 'Xor', 'NotSame']

# attributes available for given dataset (shapenet)
# need to be updated if other datasets are used
ATTRS = ['object', 'view_angle', 'category', 'location']


def compare_when(when_list):
    # input: a list of "last%d"%k
    # output: the maximum delay the task can take (max k)
    # note, n_frames = compare_when + 1; if when_list is ['last0'], then there should be 1 frame
    return max(list(map(lambda x: get_k(x), when_list)))


def get_target_value(t):
    # Convert target t to string and convert True/False target values
    # to lower case strings for consistency with other uses of true/false
    # in vocabularies.
    t = str(t)
    if t is True or t == 'True':
        return 'true'
    if t is False or t == 'False':
        return 'false'
    return t


DATA = None


# TODO: remove this global variable, and use instances of the class Data during usage

class Data:
    """
    input:
        dir_path: file path to the dataset
        max_memory: maximum frames an object can be used to solve a task
        grid_size: configuration of the canvas
        train: boolean for whether constant is for a train or val stim set
    """

    def __init__(self, dir_path=None, max_memory: int = 10, grid_size: Tuple[int, int] = (2, 2), train: bool = True):
        if dir_path is None:
            dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                    './data/min_shapenet_easy_angle')
        self.dir_path = dir_path
        self.train = train
        # TODO: remove max_memory
        self.MAX_MEMORY = max_memory

        if not os.path.exists(self.dir_path):
            raise ValueError('Data folder does not exist.')
        pkls = sorted([fname for fname in glob.glob(f'{dir_path}/**/*.pkl', recursive=True)])
        csvs = sorted([fname for fname in glob.glob(f'{dir_path}/**/*.csv', recursive=True)])
        if len(csvs) > 0:
            self.fp = csvs[0]
            self.df = pd.read_csv(self.fp)
        elif len(pkls) > 0:
            self.fp = pkls[0]
            self.df: pd.DataFrame = pd.read_pickle(self.fp)
        else:
            raise ValueError(f'No dataset meta information found in {dir_path}')

        self.MOD_DICT = self.get_mod_dict()
        self.mods_with_mapping = dict()
        if 'ctg' in self.df.columns.values:
            self.IDX2Category = dict()
            for cat, _ in self.MOD_DICT.items():
                labels = self.df[self.df['ctg_mod'] == cat]['ctg']
                assert len(set(labels)) == 1, f'found more than 1 label for the cateogry {cat}'
                self.IDX2Category[cat] = set(labels).pop()
            self.mods_with_mapping['category'] = self.IDX2Category

        self.ALLSPACES = ['left', 'right', 'top', 'bottom']
        self.ALLCATEGORIES = list(self.MOD_DICT.keys())
        self.ALLOBJECTS = {c: list(self.MOD_DICT[c].keys()) for c in self.MOD_DICT}
        self.ALLVIEWANGLES = self.MOD_DICT

        self.INVALID = 'invalid'
        self.train_image_path = None
        self.valid_image_path = None

        self.grid = get_grid(grid_size)

    def get_shapenet_object(self, obj, obj_size, training_path=None, validation_path=None):
        # sample stimuli that satisfies the properties specified by obj dictionary
        if not self.train:
            # find validation image dataset folder
            if validation_path is None:
                if self.valid_image_path is None:
                    valids = [fname for fname in glob.glob(f'{self.dir_path}/**/validation', recursive=True)]
                    if valids:
                        if os.path.isdir(valids[0]):
                            self.valid_image_path = valids[0]
                    else:
                        self.valid_image_path = self.dir_path
            else:
                self.valid_image_path = validation_path
            image_path = self.valid_image_path
        else:
            # find train image dataset folder
            if training_path is None:
                if self.train_image_path is None:
                    trains = [fname for fname in glob.glob(f'{self.dir_path}/**/train', recursive=True)]
                    if trains:
                        if os.path.isdir(trains[0]):
                            self.train_image_path = trains[0]
                    else:
                        self.train_image_path = self.dir_path
            else:
                self.train_image_path = validation_path
            image_path = self.train_image_path

        obj_cat: pd.DataFrame = self.df.loc[(self.df['ctg_mod'] == obj.category) &
                                            (self.df['obj_mod'] == obj.object) &
                                            (self.df['ang_mod'] == obj.view_angle)]
        if len(obj_cat) <= 0:
            raise ValueError(obj.category, obj.object, obj.view_angle)

        obj_ref = int(obj_cat.iloc[0]['ref'])
        obj_path = os.path.join(image_path, f'{obj_ref}/image.png')

        image = cv2.imread(obj_path)
        object_arr = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), obj_size)
        return object_arr

    def get_grid_key(self, space):
        # convert from space to grid coordinate
        return list(self.grid.keys())[list(self.grid.values()).index(list(space.value))]

    @property
    def ALLWHENS(self):
        return [f'last{k}' for k in range(self.MAX_MEMORY)]

    @property
    def ALLWHENS_PROB(self):
        # all delays have equal probability
        return [1 / (self.MAX_MEMORY)] * len(self.ALLWHENS)

    def get_mod_dict(self):
        # return an exhausitive list of all possible feature combinations
        MOD_DICT = dict()
        for i in self.df['ctg_mod'].unique():
            MOD_DICT[i] = dict()
            for cat in self.df[self.df['ctg_mod'] == i]['obj_mod'].unique():
                MOD_DICT[i][cat] = list(map(
                    int,
                    self.df[(self.df['ctg_mod'] == i) & (self.df['obj_mod'] == cat)]['ang_mod'].unique()
                ))
        return MOD_DICT


def get_k(last_k):
    k_s = list(map(int, re.findall(r'\d+', last_k)))
    assert len(k_s) == 1
    return k_s[0]


def get_grid(grid_size):
    # return (grid_size,grid_size) array of sg.space.values
    # convert from grid to space

    x_space, y_space = grid_size[0] + 1, grid_size[1] + 1
    x_coords, y_coords = np.linspace(0, 1, x_space), np.linspace(0, 1, y_space)
    xx, yy = np.meshgrid(x_coords, y_coords, sparse=True)
    xx, yy = xx.flatten(), yy.flatten()
    grid_spaces = {(i, j): [(x_i, x_k), (y_i, y_k)] for i, (x_i, x_k) in enumerate(zip(xx[0::], xx[1::])) for
                   j, (y_i, y_k) in enumerate(zip(yy[0::], yy[1::]))}
    return OrderedDict(grid_spaces)
