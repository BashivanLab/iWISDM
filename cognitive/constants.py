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

"""Store all the constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import string
import os
import numpy as np
import pandas as pd

AVG_MEM = 4
MAX_MEMORY = 4
LASTMAP = {}
for k in range(MAX_MEMORY + 1):
    LASTMAP["last%d" % k] = k

ALLWHENS = []
for k in range(MAX_MEMORY + 1):
    ALLWHENS.append("last%d" % k)
ALLWHENS_PROB = [1 / (MAX_MEMORY + 1)] * len(ALLWHENS)


def compare_when(when_list):
    return max(list(map(lambda x: LASTMAP[x], when_list)))


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


dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        'data')
if not os.path.exists(dir_path):
    print('Data folder does not exist.')

shapnet_path = os.path.join(dir_path, 'min_shapenet_easy_angle')
pickle_path = os.path.join(shapnet_path, 'train_min_shapenet_angle_easy_meta.pkl')
images_path = os.path.join(shapnet_path, 'org_shapenet/train')

df: pd.DataFrame = pd.read_pickle(pickle_path)
MOD_DICT = dict()
MOD_DICT = dict()
for i in df['ctg_mod'].unique():
    MOD_DICT[i] = dict()
    for cat in df.loc[df['ctg_mod'] == i]['obj_mod'].unique():
        MOD_DICT[i][cat] = list(df.loc[(df['ctg_mod'] == i)
                                       & (df['obj_mod'] == cat)]['ang_mod'].unique())

OBJECTPERCATEGORY = 14
CATEGORIES = 12
VIEW_ANGLES = 4
ID2Category = {i: None for i in range(CATEGORIES)}
ID2Object = {i: None for i in range(CATEGORIES * OBJECTPERCATEGORY)}
ID2ViewAngle = {i: None for i in range(VIEW_ANGLES)}

ALLSPACES = ['left', 'right', 'top', 'bottom']
ALLCATEGORIES = list(MOD_DICT.keys())
ALLOBJECTS = {c: list(MOD_DICT[c].keys()) for c in MOD_DICT}
ALLVIEWANGLES = MOD_DICT
# Comment out the following to use a smaller set of colors and shapes
# ALLCOLORS += [
#     'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige',
#     'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white']

# When the stimuli are invalid for a task
INVALID = 'invalid'

# Allowed vocabulary, the first word is invalid
# TODO: add ShapeNet vocab
INPUTVOCABULARY = [
                      'invalid',
                      '.', ',', '?',
                      'object', 'color', 'shape',
                      'loc', 'on',
                      'if', 'then', 'else',
                      'exist',
                      'equal', 'and',
                      'the', 'of', 'with',
                      'point',
                  ] + ALLSPACES + ALLCATEGORIES + ALLWHENS
# For faster str -> index lookups
INPUTVOCABULARY_DICT = dict([(k, i) for i, k in enumerate(INPUTVOCABULARY)])

INPUTVOCABULARY_SIZE = len(INPUTVOCABULARY)

OUTPUTVOCABULARY = ['true', 'false'] + ALLCATEGORIES + [ALLOBJECTS[c] for c in ALLOBJECTS]

# Maximum number of words in a sentence
MAXSEQLENGTH = 25


# If use popvec out_type
def get_prefs(grid_size):
    prefs_y, prefs_x = (np.mgrid[0:grid_size, 0:grid_size]) / (grid_size - 1.)
    prefs_x = prefs_x.flatten().astype('float32')
    prefs_y = prefs_y.flatten().astype('float32')

    # numpy array (Grid_size**2, 2)
    prefs = (np.array([prefs_x, prefs_y]).astype('float32')).T
    return prefs


GRID_SIZE = 7
PREFS = get_prefs(GRID_SIZE)

config = {'dataset': 'yang',
          'pnt_net': True,
          'in_voc_size': len(INPUTVOCABULARY),
          'grid_size': GRID_SIZE,
          'out_voc_size': len(OUTPUTVOCABULARY),
          'maxseqlength': MAXSEQLENGTH,
          'prefs': PREFS,
          }
