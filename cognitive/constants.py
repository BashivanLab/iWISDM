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
## todo: do we need this license statement?

"""Store all the constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict

# average memory duration: how many frames each object can be retained in the memory
AVG_MEM = 3
# TODO: if only 1 stim per frame, then number of selects is limited by max_memory

# define all available operators for constructing graphs
LOGIC_OPS = ['And', 'Or', 'Xor', 'IsSame']
BOOLEAN_OUT_OPS = ['IsSame', 'Exist', 'And', 'Or', 'Xor', 'NotEqual']

# attributes available for given dataset (shapenet)
# need to be updated if other datasets are used
ATTRS = ['object', 'view_angle', 'category', 'loc']


def compare_when(when_list):
    # input: a list of "last%d"%k
    # output: the maximum delay the task can take (max k)
    return max(list(map(lambda x: DATA.LASTMAP[x], when_list)))


def get_target_value(t):
    # Convert target t to string and convert True/False target values
    # to lower case strings for consistency with other uses of true/false
    # in vocabularies.
    t = t.value if hasattr(t, 'value') else str(t)
    if t is True or t == 'True':
        return 'true'
    if t is False or t == 'False':
        return 'false'
    return t


DATA = None


class Data:
    """
    input:
        dir_path: file path to the dataset
        max_memory: maximum frames an object can be used to solve a task
        grid_size: configuration of the canvas
    """
    def __init__(self, dir_path=None, max_memory=5, grid_size=[2, 2]):
        if dir_path is None:
            dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                    './data/min_shapenet_easy_angle')
        self.dir_path = dir_path

        if not os.path.exists(self.dir_path):
            print('Data folder does not exist.')
        pkls = sorted([fname for fname in glob.glob(f'{dir_path}/**/*.pkl', recursive=True)])
        # print('Stimuli Directory: ', dir_path)

        assert len(pkls) > 0
        self.pkl = pkls[0]
        self.df: pd.DataFrame = pd.read_pickle(self.pkl)
        self.MOD_DICT = dict()
        for i in self.df['ctg_mod'].unique():
            self.MOD_DICT[int(i)] = dict()
            for cat in self.df.loc[self.df['ctg_mod'] == i]['obj_mod'].unique():
                self.MOD_DICT[int(i)][int(cat)] = list(map(int, list(self.df.loc[(self.df['ctg_mod'] == i)
                                                                                 & (self.df['obj_mod'] == cat)][
                                                                         'ang_mod'].unique())))

        self.MAX_MEMORY = max_memory
        OBJECTPERCATEGORY = 14
        CATEGORIES = 12
        VIEW_ANGLES = 4
        self.ID2Category = {i: None for i in range(CATEGORIES)}
        self.ID2Object = {i: None for i in range(CATEGORIES * OBJECTPERCATEGORY)}
        self.ID2ViewAngle = {i: None for i in range(VIEW_ANGLES)}

        self.ALLSPACES = ['left', 'right', 'top', 'bottom']
        self.ALLCATEGORIES = list(self.MOD_DICT.keys())
        self.ALLOBJECTS = {c: list(self.MOD_DICT[c].keys()) for c in self.MOD_DICT}
        self.ALLVIEWANGLES = self.MOD_DICT

        self.INVALID = 'invalid'

        # Allowed vocabulary, the first word is invalid
        # TODO: add ShapeNet vocab
        self.INPUTVOCABULARY = [
                                   'invalid',
                                   '.', ',', '?',
                                   'object', 'color', 'shape',
                                   'loc', 'on',
                                   'if', 'then', 'else',
                                   'exist',
                                   'equal', 'and',
                                   'the', 'of', 'with',
                                   'point',
                               ] + self.ALLSPACES + self.ALLCATEGORIES + self.ALLWHENS
        # For faster str -> index lookups
        self.INPUTVOCABULARY_DICT = dict([(k, i) for i, k in enumerate(self.INPUTVOCABULARY)])

        self.INPUTVOCABULARY_SIZE = len(self.INPUTVOCABULARY)

        self.OUTPUTVOCABULARY = ['true', 'false'] + self.ALLCATEGORIES + [self.ALLOBJECTS[c] for c in self.ALLOBJECTS]

        self.train_image_path = None
        self.valid_image_path = None

        self.grid = get_grid(grid_size)
        # print(self.grid)

    def get_shapenet_object(self, obj, obj_size, training_path=None, validation_path=None, train=True):
        # sample stimuli that satisfies the properties specified by obj dictionary
        if not train:
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

        # obj_ref = int(obj_cat.sample(1)['ref'])
        obj_ref = int(obj_cat.iloc[0]['ref'])
        obj_path = os.path.join(image_path, f'{obj_ref}/image.png')
        img = Image.open(obj_path).convert('RGB').resize(obj_size)

        return img

    def get_grid_key(self, space):
        # convert from space to grid coordinate
        return list(self.grid.keys())[list(self.grid.values()).index(list(space._value))]

    @property
    def LASTMAP(self):
        # all possible delays
        return {f'last{k}': k for k in range(self.MAX_MEMORY + 1)}

    @property
    def ALLWHENS(self):
        # todo: is this redundant to LASTMAP? there are places refering to both but I don't see the difference between the two
        return [f'last{k}' for k in range(self.MAX_MEMORY + 1)]

    @property
    def ALLWHENS_PROB(self):
        # all delays have equal probability
        return [1 / (self.MAX_MEMORY + 1)] * len(self.ALLWHENS)


def get_mod_dict(df):
    # return an exhausitive list of all possible feature combinations
    MOD_DICT = dict()
    for i in df['ctg_mod'].unique():
        MOD_DICT[i] = dict()
        for cat in df.loc[df['ctg_mod'] == i]['obj_mod'].unique():
            MOD_DICT[i][cat] = list(df.loc[(df['ctg_mod'] == i)
                                           & (df['obj_mod'] == cat)]['ang_mod'].unique())
    return MOD_DICT


# Maximum number of words in a sentence
MAXSEQLENGTH = 25


# If use popvec out_type
def get_prefs(grid_size):
    # an alternative representation of output type for grid coordinate
    prefs_y, prefs_x = (np.mgrid[0:grid_size, 0:grid_size]) / (grid_size - 1.)
    prefs_x = prefs_x.flatten().astype('float32')
    prefs_y = prefs_y.flatten().astype('float32')

    # numpy array (Grid_size**2, 2)
    prefs = (np.array([prefs_x, prefs_y]).astype('float32')).T
    return prefs


def get_grid(grid_size):
    # return (grid_size,grid_size) array of sg.space.values
    # convert from grid to space
    # todo: difference between space and grid?
    grid_size[0], grid_size[1] = grid_size[0] + 1, grid_size[1] + 1
    x_coords, y_coords = np.linspace(0, 1, grid_size[0]), np.linspace(0, 1, grid_size[1])
    xx, yy = np.meshgrid(x_coords, y_coords, sparse=True)
    xx, yy = xx.flatten(), yy.flatten()
    grid_spaces = {(i, j): [(x_i, x_k), (y_i, y_k)] for i, (x_i, x_k) in enumerate(zip(xx[0::], xx[1::])) for
                   j, (y_i, y_k) in enumerate(zip(yy[0::], yy[1::]))}

    return OrderedDict(grid_spaces)


GRID_SIZE = 7
PREFS = get_prefs(GRID_SIZE)
