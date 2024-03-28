import os
import torch
import json

import sys
sys.path.append(sys.path[0] + '/../../iWISDM/')
# generate tasks with specified task name variable
from cognitive import info_generator as ig
from cognitive import stim_generator as sg
from cognitive import task_generator as tg



def compo_info_from_task_json(f):
    """
    read the task from json file and return the compo_info object
    """

    # load the auto task from the json dict
    with open(f, 'r') as file:
        task_dict = json.load(file)

    task_dict['operator'] = tg.load_operator_json(task_dict['operator'])
    loaded_task = tg.TemporalTask(operator = task_dict['operator'],
                                n_frames=task_dict['n_frames'],
                                first_shareable=task_dict['first_shareable'], 
                                whens = task_dict['whens'])

    fi = ig.FrameInfo(loaded_task, loaded_task.generate_objset())
    compo_info = ig.TaskInfoCompo(loaded_task, fi)
    return compo_info



def random_task_merge(compo_info_list):
    """
    merge the task info list into one task info
    """
    cur_task = compo_info_list[0]
    for task in compo_info_list[1:]:
        cur_task.merge(task)
    return cur_task


def generate_trial(compo_info, img_size, fixation_cue, output_dir, idx):
    """
    Generate one trial and write it to file 
    """
    fp = os.path.join(output_dir, 'trial' + str(idx))
    compo_info.write_trial_instance(fp, img_size, fixation_cue)



# example of generating composition of random tasks
task_json_files = ["/mnt/store1/shared/RandomTasks/tasks_loc_simple/task_0.json",
                   "/mnt/store1/shared/RandomTasks/tasks_loc_simple/task_1.json",]

compo_info_list = [compo_info_from_task_json(f) for f in task_json_files]
compo_info = random_task_merge(compo_info_list)
# generate trials
generate_trial(compo_info, 
               img_size = 224, 
               fixation_cue = True, 
               output_dir = "/mnt/store1/shared/RandomTasks/", 
               idx = 0)


