import os
import json
import shutil
import argparse
from natsort import natsorted
from collections import defaultdict

import sys
sys.path.append('../')

from wisdom import make
from wisdom import read_write
import wisdom.envs.shapenet.task_generator as tg


def create_task(env):
    _, (_, task) = env.generate_tasks()[0]
    return task

def generate_trial(env, task):
    trials = env.generate_trials(tasks=[task], mode='valid')
    imgs, _, info_dict = trials[0]
    print(info_dict)
    instructions = info_dict['instruction']
    answer = info_dict['answers']

    return imgs, instructions, answer[-1], info_dict

def store_task(task, fp):
    read_write.write_task(task, fp)

def duplicate_check(current_instructions, instruction):
    if instruction in current_instructions:
        return True
    return False

def load_stored_tasks(fp):
    ts = []
    ins = []
    for task_fp in os.listdir(fp):
        with open(os.path.join(fp, task_fp), 'r') as f:            
            task = tg.read_task(f)

            _, instructions, _, _ = generate_trial(env, task)
            ins.append(instructions)
            ts.append(task)

    return ts, ins

def create_tasks(env, track_tf, **kwargs):
    total_and = 0
    total_or = 0
    total_not = 0

    # Load tasks if they exist
    if os.listdir(kwargs['tasks_dir']) != []:
        tasks, task_ins = load_stored_tasks(kwargs['tasks_dir'])
        for k, v in track_tf.items():
            track_tf[k] = len(tasks) / len(track_tf)
    else:
        tasks = []
        task_ins = []

    while len(tasks) < kwargs['n_tasks']:
        print(len(tasks))
        print(kwargs['n_tasks'])
        task = create_task(env)
        print('task.n_frames: ', task.n_frames)
        if task.n_frames <= kwargs['max_len']:
            print('under max len')
            imgs, instructions, answer, info_dict = generate_trial(env, task)
            n_and = instructions.count(' and ')
            n_or = instructions.count(' or ')
            if kwargs['min_bool_ops'] <= (n_and + n_or) <= kwargs['max_bool_ops']:
                print('under bool op limit')
                if kwargs['force_balance']:
                    print(track_tf[answer])
                    if (track_tf[answer] + 1) / kwargs['n_trials'] <= 1 / len(track_tf):
                        print('balanced')
                        if not duplicate_check(task_ins, instructions):
                            track_tf[answer] += 1
                            total_and += n_and
                            total_or += n_or
                            total_not += instructions.count(' not ')
                            task_ins.append(instructions)
                            store_task(task, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')
                            read_write.write_trial(imgs, info_dict, os.path.join(kwargs['trials_dir'], 'trial' + str(len(tasks))))
                            tasks.append(task)
                else:
                    if not duplicate_check(task_ins, instructions):
                        track_tf[answer] += 1
                        total_and += n_and
                        total_or += n_or
                        total_not += instructions.count(' not ')
                        task_ins.append(instructions)
                        store_task(tasks, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')
                        read_write.write_trial(imgs, info_dict, os.path.join(kwargs['trials_dir'], 'trial' + str(len(tasks))))
                        tasks.append(task)

    return tasks, task_ins

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--stim_dir', type=str, default='data/shapenet_handpicked_val')
    parser.add_argument('--tasks_dir', type=str, default='benchmarking/temp/low_tasks_all')
    parser.add_argument('--trials_dir', type=str, default='benchmarking/temp/low_trials_all')
    parser.add_argument('--config_path', type=str, default='benchmarking/configs/low_complexity.json')
    parser.add_argument('--max_len', type=int, default=6)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--n_tasks', type=int, default=1000)
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--min_bool_ops', type=int, default=1)
    parser.add_argument('--max_bool_ops', type=int, default=1)
    parser.add_argument('--force_balance', action='store_true', default=False)
    parser.add_argument('--non_bool_actions', action='store_true', default=False)
    args = parser.parse_args()

    print(os.listdir(args.stim_dir))

    print(args)

    # Remake task directory
    if os.path.exists(args.tasks_dir):
        shutil.rmtree(args.tasks_dir)
    os.makedirs(args.tasks_dir)

    # Remake trials directory
    if os.path.exists(args.trials_dir):
        shutil.rmtree(args.trials_dir)
    os.makedirs(args.trials_dir)

    config = json.load(open(args.config_path))

    env = make(
        env_id='ShapeNet',
        dataset_fp=args.stim_dir
    )
    env.init_env_spec(
        max_delay=0,
        delay_prob=0.5,
        add_fixation_cue=False,
        auto_gen_config=config,
        )

    # Set balance tracking dictionary
    # CHANGE DICTIONARIES TO MATCH FEATURE LABELS OF STIMULUS SET
    n_trials = args.n_trials
    if args.features == 'all' and args.non_bool_actions:
        track_tf = {'true': 0, 'false': 0, 'benches': 0, 'boats': 0, 'cars': 0, 'chairs': 0, 'couches': 0,
                    'lighting': 0, 'planes': 0, 'tables': 0, 'bottom right': 0, 'bottom left': 0, 'top left': 0,
                    'top right': 0}
        args.n_trials = args.n_trials + (len(track_tf) - args.n_trials % len(
            track_tf))  # Makes sure n_trials is divisible by length of feature space
        args.n_tasks = args.n_trials
    elif args.features == 'category' and args.non_bool_actions:
        track_tf = {'true': 0, 'false': 0, 'benches': 0, 'boats': 0, 'cars': 0, 'chairs': 0, 'couches': 0,
                    'lighting': 0, 'planes': 0, 'tables': 0}
        args.n_trials = args.n_trials + (len(track_tf) - args.n_trials % len(track_tf))
        args.n_tasks = args.n_trials
    elif args.features == 'location' and args.non_bool_actions:
        track_tf = {'true': 0, 'false': 0, 'bottom right': 0, 'bottom left': 0, 'top left': 0, 'top right': 0}
        args.n_trials = args.n_trials + (len(track_tf) - args.n_trials % len(track_tf))
        args.n_tasks = args.n_trials
    elif args.features == 'object' and args.non_bool_actions:
        track_tf = {'true': 0, 'false': 0}
        args.n_trials = args.n_trials + (len(track_tf) - args.n_trials % len(track_tf))
        args.n_tasks = args.n_trials
    else:
        track_tf = {'true': 0, 'false': 0}

    print(track_tf)
    print(args.n_trials)
    print(n_trials)
    print('total:', len(create_tasks(env, track_tf, **vars(args))[0]))

    if args.non_bool_actions:
        number_of_files_to_delete = args.n_trials - n_trials
        delete_last_n_files(args.tasks_dir, number_of_files_to_delete)
        delete_last_n_files(args.trials_dir, number_of_files_to_delete)
