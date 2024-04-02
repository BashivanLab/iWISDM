"""
classes for registering a new environment
"""

import glob
import os
from typing import Tuple, Iterable, Dict
from collections import OrderedDict

import numpy as np
import pandas as pd


class Constant:
    """
    singleton class, constants for the environment
    """

    BOOL_OP: Iterable = None
    LOGIC_OP: Iterable = None
    ATTRS: Iterable = None

    def __init__(self, **kwargs):
        return


class EnvSpec:
    """
        Specification for any environment, including the grid size,
        maximum number of delay frames in a task, the probability of adding delay frames.

        Also includes the specification for automatic random task generation,
        this includes maximum number of operators in the task, maximum depth of the task,
        the root operator of the task, and the generation configuration.

        Args:
            grid_size:
            max_delay:
            delay_prob:
            max_op:
            max_depth:
            root_op:
            select_limit:
            auto_gen_config:
        """

    def __init__(
            self,
            grid_size: Tuple[int, int] = (2, 2),
            max_delay: int = 5,
            delay_prob=None,
            auto_gen_config: Dict = None,
            **kwargs
    ):
        self.MAX_DELAY = max_delay
        self.delay_prob = delay_prob
        self.auto_gen_config = auto_gen_config
        self.grid = get_grid(grid_size)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_grid_key(self, space):
        """
        convert from space to grid coordinate
        @param space:
        @return:
        """
        # convert from space to grid coordinate
        return list(self.grid.keys())[list(self.grid.values()).index(list(space.value))]


class StimData:
    """
    input:
        dir_path: file path to the stimuli dataset
    """

    def __init__(self, dir_path: str = None):
        self.dir_path = dir_path

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

        splits = {
            'train': '',
            'validation': '',
            'test': ''
        }
        for split in splits.keys():
            dirs = [fname for fname in glob.glob(f'{self.dir_path}/**/{split}', recursive=True)]
            if dirs:
                assert len(dirs) == 1
                if os.path.isdir(dirs[0]):
                    splits[split] = dirs[0]
        self.train_image_path = splits['train']
        self.valid_image_path = splits['validation']
        self.test_image_path = splits['test']

        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def get_object(self, obj, obj_size):
        raise NotImplementedError


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
