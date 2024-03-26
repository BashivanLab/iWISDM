import os
import re
from typing import Tuple, Dict, Iterable, List
import random
from copy import deepcopy

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from wisdom.core import Stimulus, Attribute, Task, Operator
from wisdom.envs.registration import Constant, EnvSpec, StimData
from wisdom.utils.read_write import read_img


def compare_when(when_list):
    """
    Compare the when_list to get number of frames the task can take
    @param when_list: a list of "last%d"%k
    @return: the number of frames the task can take (max k)
    """
    # note, n_frames = compare_when + 1; if when_list is ['last0'], then there should be 1 frame
    return max(list(map(lambda x: get_k(x), when_list)))


def find_delays(when_list):
    """
    Find the delay frames in the when_list
    @param when_list: a list of "last%d"
    @return: the delays in the when_list
    """
    whens = sorted(list(map(lambda x: get_k(x), when_list)))
    assert len(set(whens)) == len(whens)
    no_delay = set(range(min(whens), max(whens)))
    delays = no_delay - set(whens)
    return list(map(lambda x: f'last_{x}', sorted(delays)))


def get_k(last_k: str):
    """
    Get the integer k from the string "last%d"%k
    @param last_k: last_k string
    @return: integer k in last_k
    """
    k_s = list(map(int, re.findall(r'\d+', last_k)))
    assert len(k_s) == 1
    return k_s[0]


def get_target_value(t):
    """
    Convert boolean target value to string for task answers
    @param t: target value
    @return: the task answer format of the target value
    """
    # Convert target t to string and convert True/False target values
    # to lower case strings for consistency with other uses of true/false
    # in vocabularies.
    t = str(t)
    if t is True or t == 'True':
        return 'true'
    if t is False or t == 'False':
        return 'false'
    return t


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


class SNConst(Constant):
    """
    singleton class, constants for the ShapeNet environment
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.INVALID = 'invalid'
        # define all available operators for constructing graphs
        self.LOGIC_OPS = ['And', 'Or', 'Xor', 'IsSame', "NotSame"]
        self.BOOL_OP = ['IsSame', 'Exist', 'And', 'Or', 'Xor', 'NotSame']
        self.ALLSPACES = ['left', 'right', 'top', 'bottom']

        # attributes available for given dataset (shapenet)
        # need to be updated if other datasets are used
        self.ATTRS = ['object', 'view_angle', 'category', 'location']


class SNEnvSpec(EnvSpec):
    """
    Environment specification for ShapeNet
    """

    def __init__(
            self,
            grid_size: Tuple[int, int] = (2, 2),
            max_delay: int = 5,
            delay_prob: float = 0.5,
            auto_gen_config: Dict = None,
            add_fixation_cue: True = False,
            canvas_size: int = 224,
            **kwargs
    ):
        if auto_gen_config is None:
            auto_gen_config = {
                'op_dict': {
                    "Select":
                        {
                            "n_downstream": 4,
                            "downstream": ["GetLoc", "GetCategory", "GetObject"],
                            "same_children_op": False,
                            "min_depth": 1,
                            "min_op": 1,
                        },
                    "GetCategory":
                        {
                            "n_downstream": 1,
                            "downstream": ["Select"],
                            "min_depth": 2,
                            "min_op": 2,
                        },
                    "GetLoc":
                        {
                            "n_downstream": 1,
                            "downstream": ["Select"],
                            "min_depth": 2,
                            "min_op": 2,
                        },
                    "GetObject":
                        {
                            "n_downstream": 1,
                            "downstream": ["Select"],
                            "min_depth": 2,
                            "min_op": 2,
                        },
                    "IsSame":
                        {
                            "n_downstream": 2,
                            "downstream": ["GetLoc", "GetCategory", "GetObject", "CONST"],
                            "sample_dist": [4 / 15, 4 / 15, 4 / 15, 1 / 5],

                            "same_children_op": True,
                            "min_depth": 3,
                            "min_op": 7,
                        },
                    "NotSame":
                        {
                            "n_downstream": 2,
                            "downstream": ["GetLoc", "GetCategory", "GetObject", "CONST"],
                            "sample_dist": [4 / 15, 4 / 15, 4 / 15, 1 / 5],
                            "same_children_op": True,
                            "min_depth": 3,
                            "min_op": 7,
                        },
                    "And":
                        {
                            "n_downstream": 2,
                            "downstream": ["IsSame", "NotSame", "And", "Or"],
                            "same_children_op": False,
                            "min_depth": 4,
                            "min_op": 15,
                        },
                    "Or":
                        {
                            "n_downstream": 2,
                            "downstream": ["IsSame", "NotSame", "And", "Or"],
                            "same_children_op": False,
                            "min_depth": 4,
                            "min_op": 15,
                        },
                    "CONST":
                        {
                            "n_downstream": 0,
                            "downstream": [],
                            "sample_dist": [],
                            "same_children_op": False,
                            "min_depth": 1,
                            "min_op": 1,
                        }
                },
                # root_ops are the operators to begin a task
                'root_ops': ["IsSame", "And", "Or", "NotSame", "GetLoc", "GetCategory"],
                'boolean_ops': ["IsSame", "And", "Or", "NotSame", ],
                # all tasks end with select
                'leaf_op': ["Select"],
                'mid_op': ["Switch"],
                'max_op': 20,
                'max_depth': 10,
                'max_switch': 1,
                'switch_threshold': 0,
                'select_limit': False,
            }
        self.add_fixation_cue = add_fixation_cue
        self.canvas_size = canvas_size
        super().__init__(
            grid_size,
            max_delay,
            delay_prob,
            auto_gen_config,
            **kwargs,
        )

    def sample_when(self, n: int = 1, existing_whens=None) -> list:
        """
        sample n 'lastk' values,
        avoid sampling multiple stimuli per frame
        @param n: how many values to sample
        @param existing_whens: list of existing 'lastk' values
        @return: list of n 'lastk' values
        """
        whens = list()
        i, count, n_delays = 1, 0, 0
        if existing_whens:
            max_k = compare_when(existing_whens)
            n_delays += len(find_delays(existing_whens))
            i += max_k

        while count < n:
            add_delay = np.random.random() < self.delay_prob
            if add_delay and n_delays < self.MAX_DELAY:
                i += 1
                n_delays += 1
            else:
                whens.append(f'last{i}')
                count += 1
                i += 1
        return whens

    def check_whens(self, whens: List[str], existing_whens: List[str] = None):
        """
        Check if the whens are valid, i.e. only 1 stimulus per frame
        @param whens: the list of 'lastk' values
        @param existing_whens: existing list of 'lastk' values
        @return: resampled whens
        """
        # added check_whens to ensure 1 stimulus per frame
        existing_whens = set() if not existing_whens else set(existing_whens)
        len_ew = len(existing_whens)

        while len(set(whens) | existing_whens) != (len(whens) + len_ew):
            whens = self.sample_when(len(whens), existing_whens)
        return whens


class SNStimData(StimData):
    def __init__(self, dir_path=None):
        super().__init__(dir_path)
        self.ATTR_DICT = self.get_all_attributes()
        self.attr_with_mapping = self.get_attr_str_mapping()
        self.ALLCATEGORIES = list(self.ATTR_DICT.keys())
        self.ALLOBJECTS = {c: list(self.ATTR_DICT[c].keys()) for c in self.ATTR_DICT}
        self.ALLVIEWANGLES = self.ATTR_DICT

    def get_object(self, obj: Stimulus, obj_size: Tuple[int, int], mode: str = 'train') -> NDArray:
        """
        Get the image array of the object from the provided data directory
        @param obj: a stimulus instance with attribute values
        @param obj_size: the size of the stimulus on the canvas
        @param mode: the split, train, test, or validation
        @return: image array, RGB format
        """
        obj_pd: pd.DataFrame = self.df.loc[(self.df['ctg_mod'] == obj.category) &
                                           (self.df['obj_mod'] == obj.object) &
                                           (self.df['ang_mod'] == obj.view_angle)]
        if len(obj_pd) <= 0:
            raise ValueError(f'ShapeNet object with '
                             f'category {obj.category}, identity {obj.object}, view angle {obj.view_angle} not found')

        if mode == 'train':
            image_path = self.train_image_path
        elif mode == 'valid':
            image_path = self.valid_image_path
        elif mode == 'test':
            image_path = self.test_image_path
        else:
            raise ValueError(f'Invalid mode {mode}, only [train, valid, test]')

        obj_ref = int(obj_pd.iloc[0]['ref'])
        obj_path = os.path.join(image_path, f'{obj_ref}/image.png')

        return read_img(obj_path, obj_size, color_format='RGB')

    def get_all_attributes(self):
        """
        Get all shapenet attributes from the dataset, including the hierarchy
        @return: dictionary of the format {category: {object: [view angles]}}
        """
        ATTR_DICT = dict()
        for i in self.df['ctg_mod'].unique():
            ATTR_DICT[i] = dict()
            for cat in self.df[self.df['ctg_mod'] == i]['obj_mod'].unique():
                ATTR_DICT[i][cat] = list(map(
                    int,
                    self.df[(self.df['ctg_mod'] == i) & (self.df['obj_mod'] == cat)]['ang_mod'].unique()
                ))
        return ATTR_DICT

    def get_attr_str_mapping(self):
        """
        get the mapping from integer attribute value to string description
        e.g. 0 -> 'airplane', 1 -> 'car'
        @return: the mapping dictionary of the format {attribute: {int: str}}
        """
        self.attr_with_mapping = dict()
        if 'ctg' in self.df.columns.values:
            IDX2Category = dict()
            for cat, _ in self.ATTR_DICT.items():
                labels = self.df[self.df['ctg_mod'] == cat]['ctg']
                assert len(set(labels)) == 1, f'found more than 1 label for the cateogry {cat}'
                IDX2Category[cat] = set(labels).pop()
            self.attr_with_mapping['category'] = IDX2Category
        return self.attr_with_mapping


DATA = SNConst()
