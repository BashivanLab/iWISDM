import os
import json
import shutil
import argparse
import numpy as np
import random
from natsort import natsorted

import sys
sys.path.append('../')

from iwisdm import make
from iwisdm import read_write
import iwisdm.envs.shapenet.task_generator as tg

def create_task(env):
    _, (_, task) = env.generate_tasks()[0]

    return task

def generate_trial(env, task, mode):
    trials = env.generate_trials(tasks=[task], mode=mode)
    imgs, _, info_dict = trials[0]
    instructions = info_dict['instruction']

    return imgs, instructions, info_dict

def store_task(task, fp):
    read_write.write_task(task, fp)

def duplicate_check(tasks, task):
    tasks_len = len(tasks)
    if len(set(tasks + [task])) == tasks_len:
        return True
    return False

def load_stored_tasks(env, fp):
    ts = []
    for task_fp in os.listdir(fp):
        task = env.read_task(os.path.join(fp, task_fp))
        ts.append(task)

    return ts

def create_tasks(env, **kwargs):
    # Load tasks if they exist
    if os.listdir(kwargs['tasks_dir']) != []:
        tasks = load_stored_tasks(env, kwargs['tasks_dir'])
    else:
        tasks = []

    # Create tasks 
    while len(tasks) < kwargs['n_tasks']:
        task = create_task(env)

        # Check if task meets length requirements
        if  not (kwargs['min_len'] <= task.n_frames <= kwargs['max_len']):
            continue

        _, instructions, _ = generate_trial(env, task,
                                                                mode='train' if kwargs['train'] else 'val')
        n_and = instructions.count(' and ')
        n_or = instructions.count(' or ')
            
        # Check if task meets joint operator requirements
        if not (kwargs['min_joint_ops'] <= (n_and + n_or) <= kwargs['max_joint_ops']):
            continue
    
        n_delay = task.n_frames - instructions.count('observe')

        # Check if task meets delay frame requirements
        if not (kwargs['min_delay'] <= n_delay <= kwargs['max_delay']):
            continue

        # Check if task is a duplicate
        if not duplicate_check(tasks, task):

            store_task(task, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')

            info_dicts = []
            t_per_t = kwargs['n_trials'] // kwargs['n_tasks']
            i = t_per_t

            while i > 0:
                imgs, _, info_dict = generate_trial(env, task, mode='train' if kwargs['train'] else 'val')

                if info_dict not in info_dicts:
                    read_write.write_trial(imgs, info_dict, os.path.join(kwargs['trials_dir'], 'trial' + 
                                                                            str(len(tasks) * t_per_t + t_per_t - i)))
                    info_dicts.append(info_dict)
                    i -= 1

            print('tasks left to create:', kwargs['n_tasks'] - len(tasks))
            tasks.append(task)
        else:
            print('duplicate task')
            print('tasks left to create:', kwargs['n_tasks'] - len(tasks))
            continue
    return tasks


def delete_last_n_files(directory, n):
    files = os.listdir(directory)
    files = natsorted(files)  # Sort files in alphanumeric order
    files_to_delete = files[-n:]

    for file_to_delete in files_to_delete:
        file_path = os.path.join(directory, file_to_delete)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            shutil.rmtree(file_path)

# Shuffle files in directory 
def shuffle_files(directory):
    files = os.listdir(directory)

    # Create temp directory
    if not os.path.exists(os.path.join(directory, 'temp')):
        os.makedirs(os.path.join(directory, 'temp'))
    else:
        shutil.rmtree(os.path.join(directory, 'temp'))
        os.makedirs(os.path.join(directory, 'temp'))

    # Shuffle files
    indices = np.arange(len(files))
    np.random.shuffle(indices)

    # Copy files to temp directory with new indices
    for i, file in enumerate(files):
        shutil.copyfile(os.path.join(directory, file), os.path.join(directory, 'temp', str(indices[i]) + '.json'))

    # Copy and replace files in original directory
    for file in os.listdir(os.path.join(directory, 'temp')):
        shutil.copyfile(os.path.join(directory, 'temp', file), os.path.join(directory, file))

    # Remove temp directory
    shutil.rmtree(os.path.join(directory, 'temp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--stim_dir', type=str, default='data/shapenet_handpicked')
    parser.add_argument('--tasks_dir', type=str, default='benchmarking/tasks/test_3ff')
    parser.add_argument('--trials_dir', type=str, default='benchmarking/temp/test_3ff')
    parser.add_argument('--config_path', type=str, default='/home/lucas/projects/iWISDM/benchmarking/configs/custom_no_const/3f_thesis.json')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=3)
    parser.add_argument('--min_delay', type=int, default=0)
    parser.add_argument('--max_delay', type=int, default=0)
    parser.add_argument('--delay_prob', type=float, default=0.5)
    parser.add_argument('--dup_prob', type=float, default=0.25)
    parser.add_argument('--n_trials', type=int, default=32)
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--min_joint_ops', type=int, default=0)
    parser.add_argument('--max_joint_ops', type=int, default=2)
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()

    print(args)

    # Assert that n_trials is >= n_tasks or n_trails is 0
    assert args.n_trials >= args.n_tasks or args.n_trials == 0, 'n_trials must be >= n_tasks or 0'

    # Make task directory
    if not os.path.exists(args.tasks_dir):
        os.makedirs(args.tasks_dir)

    # Make trials directory
    if not os.path.exists(args.trials_dir):
        os.makedirs(args.trials_dir)

    # Load config
    config = json.load(open(args.config_path))
    
    # Create environment
    env = make(
        env_id='ShapeNet',
        dataset_fp=args.stim_dir
    )

    # Check if max_delay provided
    if args.max_delay == -1:
        args.max_delay = max(args.max_len - 2, 0)

    # Initialize environment
    env_spec = env.init_env_spec(
        max_delay=args.max_delay,
        delay_prob=args.delay_prob,
        add_fixation_cue=True,
        auto_gen_config=config,
        dup_prob=args.dup_prob,
    )

    env.set_env_spec(env_spec)

    try:
        new_tasks = create_tasks(env, **vars(args))
        print('tasks created:', len(new_tasks))
        print('trials created:', len(os.listdir(args.trials_dir)))
        if args.shuffle:
            print('Shuffling files...')
            shuffle_files(args.tasks_dir)
    except Exception as e:
        print('Error creating tasks:', e)
        if args.shuffle:
            print('Shuffling files...')
            shuffle_files(args.tasks_dir)
        raise e