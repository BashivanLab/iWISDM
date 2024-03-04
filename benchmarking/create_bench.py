import os
import json
import shutil
import argparse
from natsort import natsorted
from collections import defaultdict

import sys

sys.path.append(sys.path[0] + '/../../COG_v3_shapenet')

from cognitive import info_generator as ig
from cognitive import task_generator as tg
from cognitive import task_bank as tb
from cognitive.auto_task import auto_task_util as auto_task
from cognitive import constants as const


def create_task(params):
    _, task = auto_task.task_generator(params['max_switch'],
                                       params['switch_threshold'],
                                       params['max_op'],
                                       params['max_depth'],
                                       params['select_limit'])
    return task[1]


def generate_trial(task, fixation_cue=False, img_size=224):
    # Creates FrameInfo and TaskInfo objects for the task
    frame_info = ig.FrameInfo(task, task.generate_objset())
    compo_info = ig.TaskInfoCompo(task, frame_info)

    _, instructions, answer = compo_info.generate_trial(img_size, fixation_cue)

    return instructions, answer[-1], compo_info


def store_task(task, fp):
    task.to_json(fp)


def duplicate_check(current_instructions, instruction):
    if instruction in current_instructions:
        return True
    return False


def load_stored_tasks(fp):
    ts = []
    ins = []
    for task_fp in os.listdir(fp):
        with open(os.path.join(fp, task_fp), 'r') as f:
            task_dict = json.load(f)
            # first you have to load the operator objects
            task_dict['operator'] = tg.load_operator_json(task_dict['operator'])

            # we must reinitialize using the parent task class. (the created task object is functionally identical) 
            task = tg.TemporalTask(
                operator=task_dict['operator'],
                n_frames=task_dict['n_frames'],
                first_shareable=task_dict['first_shareable'],
                whens=task_dict['whens']
            )

            instructions = generate_trial(task)
            ins.append(instructions)
            ts.append(task)

    return ts, ins


def create_tasks(track_tf, task_params, **kwargs):
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
        task = create_task(task_params)
        if task.n_frames <= kwargs['max_len']:
            print('under max len')
            instructions, answer, compo_info = generate_trial(task)
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
                            compo_info.write_trial_instance(
                                os.path.join(kwargs['trials_dir'], 'trial' + str(len(tasks))), 224, kwargs['train'])
                            tasks.append(task)
                else:
                    if not duplicate_check(task_ins, instructions):
                        track_tf[answer] += 1
                        total_and += n_and
                        total_or += n_or
                        total_not += instructions.count(' not ')
                        task_ins.append(instructions)
                        store_task(task, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')
                        compo_info.write_trial_instance(os.path.join(kwargs['trials_dir'], 'trial' + str(len(tasks))),
                                                        224, kwargs['train'])
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
    parser.add_argument('--max_memory', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=6)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--n_tasks', type=int, default=1000)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--max_op', type=int, default=15)
    parser.add_argument('--max_switch', type=int, default=0)
    parser.add_argument('--switch_threshold', type=float, default=1.0)
    parser.add_argument('--select_limit', action='store_true', default=False)
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--min_bool_ops', type=int, default=1)
    parser.add_argument('--max_bool_ops', type=int, default=1)
    parser.add_argument('--force_balance', action='store_true', default=False)
    parser.add_argument('--non_bool_actions', action='store_true', default=False)
    args = parser.parse_args()

    print(args)

    # Remake task directory
    if os.path.exists(args.tasks_dir):
        shutil.rmtree(args.tasks_dir)
    os.makedirs(args.tasks_dir)
    # Remake trials directory
    if os.path.exists(args.trials_dir):
        shutil.rmtree(args.trials_dir)
    os.makedirs(args.trials_dir)


    task_params = {
        'max_op': args.max_op,
        'max_depth': args.max_memory,
        'max_switch': args.max_switch,
        'select_limit': args.select_limit,
        'switch_threshold': args.switch_threshold
    }

    const.DATA = const.Data(
        dir_path=args.stim_dir,
        max_memory=args.max_memory,
        train=False
    )
    with open(args.config_path) as f:
        config = json.load(f)
        op_dict = config['op_dict']
        root_ops = config['root_ops']
        boolean_ops = config['boolean_ops']

        # For all features
        # op_dict['IsSame']['sample_dist'] = [4 / 15, 4 / 15, 4 / 15, 1 / 5]
        # op_dict['NotSame']['sample_dist'] = [4 / 15, 4 / 15, 4 / 15, 1 / 5]

        # For loc or cat features only
        op_dict['IsSame']['sample_dist'] = [0.9, 0.1]
        op_dict['NotSame']['sample_dist'] = [0.9, 0.1]

        auto_task.root_ops = root_ops
        auto_task.boolean_ops = boolean_ops
        auto_task.op_dict = defaultdict(dict, **op_dict)
        auto_task.op_depth_limit = {k: v['min_depth'] for k, v in auto_task.op_dict.items()}
        auto_task.op_operators_limit = {k: v['min_op'] for k, v in auto_task.op_dict.items()}

    # Set balance tracking dictionary
    # CHANGE DICTIONARIES TO MATCH FEATURE LABELS OF STIMULUS SET
    n_trials = args.n_trials
    if args.features == 'all' and args.non_bool_actions:
        track_tf = {'true': 0, 'false': 0, 'benches': 0, 'boats': 0, 'cars': 0, 'chairs': 0, 'couches': 0,
                    'lighting': 0, 'planes': 0, 'tables': 0, 'bottom right': 0, 'bottom left': 0, 'top left': 0,
                    'top right': 0}
        args.n_trials = args.n_trials + (len(track_tf) - args.n_trials % len(track_tf))  # Makes sure n_trials is divisible by length of feature space
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
    print('total:', len(create_tasks(track_tf, task_params, **vars(args))[0]))

    if args.non_bool_actions:
        number_of_files_to_delete = args.n_trials - n_trials
        delete_last_n_files(args.tasks_dir, number_of_files_to_delete)
        delete_last_n_files(args.trials_dir, number_of_files_to_delete)
