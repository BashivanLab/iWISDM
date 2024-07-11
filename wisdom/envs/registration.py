"""
classes for registering a new environment
"""

import glob
import os
from typing import Tuple, Iterable, Dict
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray
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

        see wisdom/envs/shapenet/registration.py for example
        Args:
            grid_size: convert image canvas into (n, n) grid for location tasks
            max_delay: upper bound on the number of delay frames in a task
            delay_prob: how likely a delay frame is added, [0.0, 1.0]
            auto_gen_config: AutoTask generation configuration, see wisdom/envs/shapenet/registration.py for example
        """

    def __init__(
            self,
            grid_size: Tuple[int, int] = (2, 2),
            max_delay: int = 5,
            delay_prob=None,
            auto_gen_config: Dict = None,
            **kwargs
    ):
        """
        environment trial generation specifications

        @param grid_size: convert image canvas into (n, n) grid for location tasks
        @param max_delay: upper bound on the number of delay frames in a task
        @param delay_prob: how likely a delay frame is added, [0.0, 1.0]
        @param auto_gen_config:
        @param kwargs:
        """
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
    class for storing and retrieving the stimuli dataset information
    Args:
        dir_path: file path to the stimuli dataset. the folder has subdir structure:
            dir_path:
                -train
                    -imgs
                    -train_metadata.{csv/pkl}
                -val
                    -imgs
                    -val.{csv/pkl}
                -test
                    -imgs
                    -test_metadata.{csv/pkl}
        find_subdir: boolean, if True, find the subdirectories for train, validation, test splits
        df: pandas dataframe, the metadata for the stimuli dataset
        img_folder_path: file path to the stimuli images
        splits: dictionary containing information to the train, validation, test splits
    """

    def __init__(
            self,
            dir_path: str = None,
            find_subdir: bool = True,
            df: pd.DataFrame = None,
            img_folder_path: str = None,
            splits: Dict = None
    ):
        self.dir_path = dir_path
        self.find_subdir = find_subdir
        if not find_subdir:
            assert img_folder_path is not None and df is not None, \
                'img_folder_path and df must be specified if find_subdir is False'
            self.img_folder_path = img_folder_path
            self.df = df
            self.splits = splits
        else:
            self.pkls = sorted([fname for fname in glob.glob(f'{dir_path}/**/*.pkl', recursive=True)])
            self.csvs = sorted([fname for fname in glob.glob(f'{dir_path}/**/*.csv', recursive=True)])

            if not os.path.exists(self.dir_path):
                raise ValueError('Data folder does not exist.')
            if not self.csvs and not self.pkls:
                raise ValueError(f'No dataset meta information found in {dir_path}')

            if df is not None:
                self.df = df
            else:
                if self.csvs:
                    fp = self.csvs[0]
                    self.df = pd.read_csv(fp)
                elif self.pkls:
                    fp = self.pkls[0]
                    self.df: pd.DataFrame = pd.read_pickle(fp)
                if not self.csvs and not self.pkls:
                    raise ValueError(f'No dataset meta information found in {dir_path}')

            if splits is not None:
                self.splits = splits
            else:
                self.splits = self.find_dataset_splits()

            self.img_folder_path = None
            if img_folder_path is not None:
                self.img_folder_path = img_folder_path
            else:
                for k, v in self.splits.items():
                    if 'path' in v:
                        if v['path']:
                            self.img_folder_path = v['path']
                            break
                if not self.img_folder_path:
                    raise ValueError('No image folder path found')

    def find_dataset_splits(self):
        assert self.find_subdir, 'find_subdir must be True to find dataset splits'
        splits = {
            'train': dict(),
            'val': dict(),
            'test': dict()
        }
        no_datafolder = True
        for split in splits.keys():
            dirs = [fname for fname in glob.glob(f'{self.dir_path}/**/{split}', recursive=True)]
            if dirs:
                no_datafolder = False
                assert len(dirs) == 1, f'found more than 1 folder for {split} split'
                if os.path.isdir(dirs[0]):
                    splits[split]['path'] = dirs[0]
            if self.csvs:
                df_fp = [fp for fp in self.csvs if split in fp.split('/')[-1]]
                if df_fp:
                    splits[split]['df'] = pd.read_csv(df_fp[0])
            elif self.pkls:
                df_fp = [fp for fp in self.pkls if split in fp.split('/')[-1]]
                if df_fp:
                    splits[split]['df'] = pd.read_pickle(df_fp[0])
        if no_datafolder:
            raise ValueError(f'No dataset splits found in data folder {self.dir_path}')
        return splits

    def get_object(self, obj, obj_size: Tuple[int, int], mode: str) -> NDArray:
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
