from typing import Tuple, List, Dict

import numpy as np

from wisdom.core import Env
import wisdom.envs.shapenet.auto_task_gen as atg
import wisdom.envs.shapenet.stim_generator as sg
import wisdom.envs.shapenet.task_generator as tg
import wisdom.envs.shapenet.info_generator as ig

import wisdom.envs.shapenet.registration as const
from wisdom.envs.shapenet.registration import SNEnvSpec, SNStimData


class ShapeNetEnv(Env):
    def __init__(self, stim_data: SNStimData, env_spec: SNEnvSpec):
        super().__init__()
        self.base_classes = [sg.SNAttribute, sg.SNStimulus, sg.ObjectSet, tg.SNTask, tg.SNOperator]

        self.stim_data = stim_data
        self.env_spec = env_spec
        self.task_gen_config = env_spec.auto_gen_config
        self.constants = const.DATA
        self.task_gen = atg.SNTaskGenerator(env_spec)
        self.reset_env()

        return

    @staticmethod
    def init_stim_data(dataset_fp: str):
        stim_data = SNStimData(dataset_fp)
        return stim_data

    @staticmethod
    def init_env_spec(**kwargs):
        env_spec = SNEnvSpec(**kwargs)
        return env_spec

    def generate_tasks(self, n: int = 1, save_fp: str = None):
        self.reset_env()
        tasks = [
            self.task_gen.generate_task()
            for _ in range(n)
        ]
        return tasks

    def generate_trials(self, tasks) -> Tuple[List[np.ndarray], List[Dict], Dict]:
        self.reset_env()

        return

    def render_trials(self):
        self.reset_env()

        return

    def reset_env(self) -> None:
        """
        Reset the environment by resetting stim_data and env_spec in base classes
        so that when env.generate_tasks(), env.generate_trials() is called,
        the tasks are generated based on self.stim_data and self.env_spec
        """
        for base_class in self.base_classes:
            base_class.stim_data = self.stim_data
            base_class.env_spec = self.env_spec
