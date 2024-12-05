import random
from typing import Tuple, List, Dict, Iterable, Union

import networkx as nx
import numpy as np

from wisdom.core import Env, Operator, Attribute, Task
import wisdom.envs.shapenet.auto_task_gen as atg
import wisdom.envs.shapenet.stim_generator as sg
import wisdom.envs.shapenet.task_generator as tg
import wisdom.envs.shapenet.info_generator as ig

import wisdom.envs.shapenet.registration as env_reg
from wisdom.envs.shapenet.registration import SNEnvSpec, SNStimData
from wisdom.envs.shapenet.task_bank import task_family_dict

GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]
TASK = Tuple[Union[Operator, Attribute], Task]


class ShapeNetEnv(Env):
    def __init__(self, stim_data: SNStimData, env_spec: SNEnvSpec):
        super().__init__()
        self.base_classes = [sg.SNAttribute, sg.SNStimulus, sg.ObjectSet, tg.SNTask, tg.SNOperator]
        self.constants = env_reg.DATA

        self.stim_data = stim_data

        self.env_spec = env_spec
        self.task_gen_config = env_spec.auto_gen_config
        self.task_gen = atg.SNTaskGenerator(env_spec)
        self.reset_env()

        self.cached_tasks = set()
        return

    @staticmethod
    def init_stim_data(dataset_fp: str, **kwargs):
        stim_data = SNStimData(dataset_fp)
        return stim_data

    @staticmethod
    def init_env_spec(**kwargs):
        env_spec = SNEnvSpec(**kwargs)
        return env_spec

    def set_env_spec(self, env_spec: SNEnvSpec):
        self.env_spec = env_spec
        self.task_gen_config = env_spec.auto_gen_config
        self.task_gen = atg.SNTaskGenerator(env_spec)
        self.reset_env()
        return

    def set_stim_data(self, stim_data: SNStimData):
        self.stim_data = stim_data
        self.reset_env()
        return

    def generate_tasks(self, n: int = 1, **kwargs) -> List[Tuple[GRAPH_TUPLE, TASK]]:
        self.reset_env()
        tasks = [
            self.task_gen.generate_task()
            for _ in range(n)
        ]
        self.cached_tasks.update([t[1][1] for t in tasks])
        return tasks

    @staticmethod
    def init_compositional_tasks(
            tasks: Iterable[tg.TemporalTask],
            task_objsets: Iterable[sg.ObjectSet] = None,
    ) -> List[ig.TaskInfoCompo]:
        if task_objsets is not None:
            assert len(tasks) == len(task_objsets)
        else:
            task_objsets = [t.generate_objset() for t in tasks]

        compo_tasks = list()
        for task, objset in zip(tasks, task_objsets):
            fi = ig.FrameInfo(task, objset)
            compo_info = ig.TaskInfoCompo(task, fi)
            compo_tasks.append(compo_info)
        return compo_tasks

    def merge_tasks(
            self,
            tasks: Iterable[tg.TemporalTask] = None,
            task_objsets: Iterable[sg.ObjectSet] = None,
            compositional_infos: Iterable[ig.TaskInfoCompo] = None,
            num_merge: int = 1,
    ) -> ig.TaskInfoCompo:
        """
        keep adding 1 random task to the compositional task
        """
        if not tasks:
            tasks = self.cached_tasks
        if not compositional_infos:
            compositional_infos = self.init_compositional_tasks(tasks, task_objsets)

        tmp = random.choice(compositional_infos)
        compositional_task = self.init_compositional_tasks(
            [tmp.tasks[0].copy()]
        )[0]
        for _ in range(num_merge):
            compo_info = random.choice(compositional_infos)
            tmp_info = self.init_compositional_tasks(
                [compo_info.tasks[0].copy()]
            )[0]
            compositional_task.merge(tmp_info)
        del tmp, tmp_info
        return compositional_task

    def generate_trials(
            self,
            tasks: Iterable[tg.TemporalTask] = None,
            task_objsets: Iterable[sg.ObjectSet] = None,
            compositional_infos: Iterable[ig.TaskInfoCompo] = None,
            mode: str = None,
            return_objset: bool = False,
            **kwargs
    ) -> List[Tuple[List[np.ndarray], List[Dict], Dict]]:
        self.reset_env(mode)
        if mode and self.stim_data.splits[mode]:
            stim_data = self.stim_data.splits[mode]['data']
        else:
            stim_data = self.stim_data

        if not compositional_infos:
            if not tasks:
                tasks = self.cached_tasks
            compositional_infos = self.init_compositional_tasks(tasks, task_objsets)

        trials = list()
        for compo_info in compositional_infos:
            if return_objset:
                imgs, per_task_info, compo_info_dict, objset = compo_info.generate_trial(
                    self.env_spec.canvas_size,
                    self.env_spec.add_fixation_cue,
                    stim_data,
                    return_objset,
                    **kwargs
                )
                trials.append((imgs, per_task_info, compo_info_dict, objset))
            else:
                imgs, per_task_info, compo_info_dict = compo_info.generate_trial(
                    self.env_spec.canvas_size,
                    self.env_spec.add_fixation_cue,
                    stim_data,
                    **kwargs
                )
                trials.append((imgs, per_task_info, compo_info_dict))
        return trials

    def render_trials(self):
        self.reset_env()

        return

    def reset_env(self, mode: str = None) -> None:
        """
        Reset the environment by resetting stim_data and env_spec in base classes
        so that when env.generate_tasks(), env.generate_trials() is called,
        the tasks are generated based on self.stim_data and self.env_spec
        """
        if mode is not None and self.stim_data.splits[mode]:
            stim_data = self.stim_data.splits[mode]['data']
        else:
            stim_data = self.stim_data

        for base_class in self.base_classes:
            base_class._stim_data = stim_data
            base_class.env_spec = self.env_spec

    def get_premade_task(self, task_family: List[str] = None, **kwargs) -> tg.TemporalTask:
        """Return a random question from the task family."""
        if task_family is None:
            task_family = list(task_family_dict.keys())
        random_task = random.choice(task_family)
        whens = self.env_spec.sample_when(2)
        return task_family_dict[random_task](whens=whens, **kwargs)
