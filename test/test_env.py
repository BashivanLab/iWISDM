import json
import shutil
import unittest

from wisdom import make
from wisdom.envs.shapenet import stim_generator as sg
from wisdom import read_write

import os


class MyTestCase(unittest.TestCase):
    def setUp(self):
        with open('../benchmarking/configs/high_complexity_all.json', 'r') as f:
            config = json.load(f)
        high_env = make(
            env_id='ShapeNet',
            dataset_fp='../data/shapenet_handpicked'
        )
        high_env.set_env_spec(high_env.init_env_spec(
            max_delay=4,
            delay_prob=0.5,
            add_fixation_cue=True,
            auto_gen_config=config,
        ))
        self.high_env = high_env

    def test_env(self):
        env = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked_val'
        )
        print(env.env_spec.auto_gen_config)
        tasks = env.generate_tasks(100)
        for t in tasks:
            _, (_, temporal_task) = t
            for i in range(100):
                trials = env.generate_trials(tasks=[temporal_task])
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'output/trial_{i}')
        # TODO: stimuli sampled from dataset splits, have 3 separate df files?
        #  self.stim_data.train = SNStimData(), etc
        env.get_premade_task()
        return

    def test_stim_data(self):
        env = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked'
        )
        for _ in range(10):
            stub_obj = sg.Object(when='last0')
            stub_obj.epoch = [1, 2]
            stub_obj.change_view_angle(sg.random_attr('view_angle'))
            train_arr = env.stim_data.get_object(stub_obj.to_static()[0], obj_size=(12, 12), mode='train')
            val_arr = env.stim_data.get_object(stub_obj.to_static()[0], obj_size=(12, 12), mode='val')
            self.assertFalse((train_arr == val_arr).all())

    def test_configs(self):
        config_1 = {
            "op_dict": {
                "Select":
                    {
                        "n_downstream": 4,
                        "downstream": ["GetCategory"],
                        "same_children_op": True,
                        "min_depth": 1,
                        "min_op": 1
                    },
                "GetCategory":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetLoc":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetObject":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "IsSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetCategory", "CONST"],

                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "NotSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetCategory", "CONST"],
                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "And":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "Or":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "CONST":
                    {
                        "n_downstream": 0,
                        "downstream": [],
                        "sample_dist": [],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    }
            },
            "root_ops": ["IsSame", "And", "Or", "NotSame", "GetCategory"],
            "boolean_ops": ["IsSame", "And", "Or", "NotSame"],
            "leaf_op": ["Select"],
            "mid_op": ["Switch"],
            "max_op": 15,
            "max_depth": 3,
            "max_switch": 0,
            "switch_threshold": 0,
            "select_limit": True
        }
        config_2 = {
            "op_dict": {
                "Select":
                    {
                        "n_downstream": 4,
                        "downstream": ["GetLoc"],
                        "same_children_op": True,
                        "min_depth": 1,
                        "min_op": 1
                    },
                "GetCategory":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetLoc":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetObject":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "IsSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],

                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "NotSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],
                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "And":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "Or":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "CONST":
                    {
                        "n_downstream": 0,
                        "downstream": [],
                        "sample_dist": [],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    }
            },
            "root_ops": ["IsSame", "And", "Or", "NotSame"],
            "boolean_ops": ["IsSame", "And", "Or", "NotSame"],
            "leaf_op": ["Select"],
            "mid_op": ["Switch"],
            "max_op": 15,
            "max_depth": 3,
            "max_switch": 0,
            "switch_threshold": 0,
            "select_limit": True
        }
        env_1 = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked_val'
        )
        env_1.set_env_spec(env_1.init_env_spec(
            max_delay=4,
            delay_prob=0.5,
            add_fixation_cue=True,
            auto_gen_config=config_1,
        ))
        env_2 = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked_val'
        )
        env_2.set_env_spec(env_2.init_env_spec(
            max_delay=5,
            delay_prob=0.5,
            add_fixation_cue=False,
            auto_gen_config=config_2,
        ))
        tasks = env_1.generate_tasks(100)
        for t in tasks:
            _, (_, temporal_task) = t
            for i in range(10):
                trials = env_1.generate_trials(tasks=[temporal_task])
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'output/trial_{i}')
        tasks = env_2.generate_tasks(100)
        for t in tasks:
            _, (_, temporal_task) = t
            for i in range(10):
                trials = env_2.generate_trials(tasks=[temporal_task])
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'output/trial_{i}')

    def test_one_frame(self):
        config_1 = {
            "op_dict": {
                "Select":
                    {
                        "n_downstream": 4,
                        "downstream": ["GetLoc", "GetCategory", "GetObject", None],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    },
                "GetCategory":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetLoc":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetObject":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "IsSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],
                        "sample_dist": [0.00000000001, 0.99999999999],  # hack
                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "NotSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],
                        "sample_dist": [0.00000000001, 0.99999999999],  # hack
                        "same_children_op": False,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "And":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "Or":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "CONST":
                    {
                        "n_downstream": 0,
                        "downstream": [],
                        "sample_dist": [],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    }
            },
            "root_ops": ["IsSame", "NotSame"],
            "boolean_ops": ["IsSame", "And", "Or", "NotSame"],
            "leaf_op": ["Select"],
            "mid_op": ["Switch"],
            "max_op": 15,
            "max_depth": 3,
            "max_switch": 0,
            "switch_threshold": 0,
            "select_limit": True
        }
        env_1 = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked_val'
        )
        env_spec = env_1.init_env_spec(
            max_delay=0,
            delay_prob=0,
            add_fixation_cue=True,
            auto_gen_config=config_1,
        )
        env_1.set_env_spec(env_spec)
        tasks = env_1.generate_tasks(100)
        for j, t in enumerate(tasks):
            _, (_, temporal_task) = t
            task_dir = f'output/task_{j}'
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
            os.makedirs(task_dir)
            for i in range(10):
                trials = env_1.generate_trials(tasks=[temporal_task])
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'{task_dir}/trial_{i}')

    def test_eq(self):
        config_1 = {
            "op_dict": {
                "Select":
                    {
                        "n_downstream": 4,
                        "downstream": ["GetLoc", "GetCategory", "GetObject", None],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    },
                "GetCategory":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetLoc":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "GetObject":
                    {
                        "n_downstream": 1,
                        "downstream": ["Select"],
                        "min_depth": 2,
                        "min_op": 2
                    },
                "IsSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],
                        "sample_dist": [0.00000000001, 0.99999999999],  # hack
                        "same_children_op": True,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "NotSame":
                    {
                        "n_downstream": 2,
                        "downstream": ["GetLoc", "CONST"],
                        "sample_dist": [0.00000000001, 0.99999999999],  # hack
                        "same_children_op": False,
                        "min_depth": 3,
                        "min_op": 7
                    },
                "And":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "Or":
                    {
                        "n_downstream": 2,
                        "downstream": ["IsSame", "NotSame", "And", "Or"],
                        "same_children_op": False,
                        "min_depth": 4,
                        "min_op": 15
                    },
                "CONST":
                    {
                        "n_downstream": 0,
                        "downstream": [],
                        "sample_dist": [],
                        "same_children_op": False,
                        "min_depth": 1,
                        "min_op": 1
                    }
            },
            "root_ops": ["IsSame", "NotSame"],
            "boolean_ops": ["IsSame", "And", "Or", "NotSame"],
            "leaf_op": ["Select"],
            "mid_op": ["Switch"],
            "max_op": 15,
            "max_depth": 3,
            "max_switch": 0,
            "switch_threshold": 0,
            "select_limit": True
        }
        env_1 = make(
            env_id='ShapeNet',
            dataset_fp='/Users/markbai/Documents/COG_v3_shapenet/data/shapenet_handpicked_val'
        )
        env_spec = env_1.init_env_spec(
            max_delay=0,
            delay_prob=0,
            add_fixation_cue=True,
            auto_gen_config=config_1,
        )
        env_1.set_env_spec(env_spec)
        tasks = env_1.generate_tasks(100)
        ts = [t[1][1] for t in tasks]
        ts_set = set(ts)
        test = ts_set.pop()
        assert (all(t != test for t in ts_set))
        assert (all(str(t) != (test) for t in ts_set))

    def test_random(self):
        tasks = self.high_env.generate_tasks(100)
        ts = [t[1][1] for t in tasks]
        ts_set = set(ts)
        test = ts_set.pop()
        assert (all(t != test for t in ts_set))
        assert (all(str(t) != (test) for t in ts_set))
        for t in tasks:
            _, (_, temporal_task) = t
            for i in range(10):
                trials = self.high_env.generate_trials(tasks=[temporal_task])
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'output/trial_{i}')
        return

    def test_merge(self):
        tasks = self.high_env.generate_tasks(100)
        for i in range(10):  # test 10 times
            t = self.high_env.merge_tasks(
                tasks=[t[1][1] for t in tasks],
                num_merge=10
            )
            trials = self.high_env.generate_trials(
                compositional_infos=[t]
            )
            imgs, _, info_dict = trials[0]
            read_write.write_trial(imgs, info_dict, f'output/trial_{i}')

    def test_distractors(self):
        tasks = self.high_env.generate_tasks(100)
        for t in tasks:
            _, (_, temporal_task) = t
            for i in range(10):
                trials = self.high_env.generate_trials(
                    tasks=[temporal_task],
                    add_distractor_time=2
                )
                imgs, _, info_dict = trials[0]
                read_write.write_trial(imgs, info_dict, f'output/trial_{i}')


if __name__ == '__main__':
    unittest.main()
