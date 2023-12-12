import sys
# Parallelism
from multiprocessing import Pool
sys.path.append('../')
import networkx as nx

import json
import pickle
import numpy as np
from cognitive import constants as const
import cognitive.task_generator as tg
from cognitive.auto_task import auto_task_util as auto_task
import os
import shutil

# todo: where to cut the number of delay frames? 
# todo: I cannot save and load the task properly [pickle load: RecursionError: Maximum recursion depth exceeded] => solved 

# todo: add other possible boolean operators for the task ["XOR", "OR"]
# todo: the boolean operator specified here is useless
# todo: parallel generaiton cause maximum recurssion error
# todo: need to check why long sequence length cause key error in key maps [related to delay frame question]


class RandomTaskParallelGen(object):
    def __init__(self, n_tasks: int, max_op: int, max_depth: int, max_switch: int, 
                 stim_dir, output_dir, select_limit=False, switch_threshold=0.3, 
                 boolean_ops=["Exist", "IsSame", "And", "NotSame"], img_size=224, 
                 train=True):

        const.DATA = const.Data(dir_path=stim_dir)
        self.n_tasks = n_tasks
        self.max_depth = max_depth
        self.max_op = max_op
        self.max_switch = max_switch
        self.select_limit = select_limit
        self.switch_threshold = switch_threshold
        self.boolean_ops = boolean_ops

        self.op_dict = auto_task.op_dict

        self.stim_dir = stim_dir
        self.output_dir = output_dir

        self.img_size = img_size
        self.output_dir = output_dir

        self.count_exit_tasks()
        self.reset() # fix the tasks to be generated
        self.iter_generate_trial() # exhaustively generate one example trial for all tasks and find max_len for future padding use
        self.cur_idx = 0

        # self.max_len= ()

    def count_exit_tasks(self):
        ## todo: xlei: remove the following line
        try:
            shutil.rmtree(os.path.join(self.output_dir, "tasks"))
        except:
            pass
        if not os.path.exists(os.path.join(self.output_dir, "tasks")):
            # If not, create it
            os.makedirs(os.path.join(self.output_dir, "tasks"))
        self.existing_tasks = len([f for f in os.listdir(os.path.join(self.output_dir, "tasks")) if os.path.isfile(os.path.join(self.output_dir, "tasks",f)) and f.startswith("task")])
        

    # Resets new set of tasks
    def reset(self):
        self.tasks = []

        for i in range(self.existing_tasks):
            if i <= self.n_tasks:
                
                f = open(os.path.join(self.output_dir, "tasks", "task_%d.json" % (i)))
                task_dict = json.load(f)
                # first you have to load the operator objects
                task_dict['operator'] = tg.load_operator_json(task_dict['operator'])

                # we must reinitialize using the parent task class. (the created task object is functionally identical) 
                updated_task = tg.TemporalTask(
                    operator=task_dict['operator'],
                    n_frames=task_dict['n_frames'],
                    first_shareable=task_dict['first_shareable'],
                    whens=task_dict['whens']
                )
                self.tasks.append(updated_task)

        
        if self.existing_tasks < self.n_tasks: # generate new tasks
            for i in range(self.n_tasks - self.existing_tasks):

                task_graph, task = auto_task.task_generator(self.max_switch,
                                                            self.switch_threshold,
                                                            self.max_op,
                                                            self.max_depth,
                                                            self.select_limit)

                task[1].to_json(os.path.join(self.output_dir, "tasks", "task_%d.json" % (self.existing_tasks + i)))
                self.tasks.append(task[1])

    # Spawns the generation of n trials of the task into seperate cores 
    def generate_trials(self, n_trials):
        with Pool() as pool:
            pool.map(self.generate_trial, range(self.cur_idx, self.cur_idx + n_trials))
        self.cur_idx += n_trials
    
    # iteratively generate trials to find max_len
    def iter_generate_trial(self):
        self.max_len = 0
        # xlei: instruction depend on each trial
        # self.instructions = []
        for i, task in enumerate(self.tasks):
            fp = os.path.join(output_dir, "trials", 'trial' + str(-1*i))
            auto_task.write_trial_instance(task, fp, self.img_size, True) # fixation_cue is on
            n_frames = len([f for f in os.listdir(fp) if os.path.isfile(os.path.join(fp,f)) and f.startswith("epoch")])
            if n_frames > self.max_len:
                self.max_len = n_frames
            # self.instructions.append(json.load(open(os.path.join(fp, "trial_info")))["instruction"])


    # Generate one trial and write it to file 
    def generate_trial(self, idx):
        index = np.random.randint(len(self.tasks))
        task = self.tasks[index]
        fp = os.path.join(output_dir, "trials", 'trial' + str(idx))
        auto_task.write_trial_instance(task, fp, self.img_size, fixation_cue = True, is_instruction = True, external_instruction = None) # fixation_cue is on

       
output_dir = '/mnt/store1/shared/RandomTasks' # the output directory
stim_dir = '/mnt/store1/shared/XLshared_large_files/new_shapenet_train' # stimulus set
n_tasks = 1 # number of tasks to be generated
max_op = 4
max_depth = 4
max_switch = 2

# this is what I mean by only generate one tasks
for n_task in range(1,n_tasks+1):
    dset = RandomTaskParallelGen(n_tasks = n_task, max_op = max_op, max_depth = max_depth, max_switch = max_switch, 
                    stim_dir = stim_dir, output_dir = output_dir, img_size=224, train=True)

# for i in range(10):
dset.generate_trials(5)
