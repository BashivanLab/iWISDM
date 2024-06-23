import os
import json
import shutil
import argparse
from natsort import natsorted

import sys

sys.path.append('../')

from wisdom import make
from wisdom import read_write
import wisdom.envs.shapenet.task_generator as tg

def create_task(env):
    _, (_, task) = env.generate_tasks()[0]

    return task

def generate_trial(env, task, mode):
    trials = env.generate_trials(tasks=[task], mode=mode,)
    imgs, _, info_dict = trials[0]
    instructions = info_dict['instruction']
    answer = info_dict['answers']

    return imgs, instructions, answer[-1], info_dict

def store_task(task, fp):
    read_write.write_task(task, fp)

def duplicate_check(current_instructions, instruction):
    if instruction in current_instructions:
        return True
    return False

def load_stored_tasks(fp, mode):
    ts = []
    ins = []
    anss = []

    for task_fp in os.listdir(fp):
        task = tg.read_task(os.path.join(fp, task_fp))

        _, instructions, answer, _ = generate_trial(env, task, mode)
        ins.append(instructions)
        ts.append(task)
        anss.append(answer)

    return ts, ins, anss

def create_tasks(env, track_tf, **kwargs):
    total_and = 0
    total_or = 0
    total_not = 0

    # Load tasks if they exist
    if os.listdir(kwargs['tasks_dir']) != []:
        tasks, task_ins, answers = load_stored_tasks(kwargs['tasks_dir'], mode='train' if kwargs['train'] else 'val')
        for ins, task, ans in zip(task_ins, tasks, answers):
            total_and += ins.count(' and ')
            total_or += ins.count(' or ')
            total_not += ins.count(' not ')
            track_tf[ans] += 1
    else:
        tasks = []
        task_ins = []

    # Create tasks 
    while len(tasks) < kwargs['n_tasks']:
        print(len(tasks))
        print(kwargs['n_tasks'])

        task = create_task(env)

        print('task.n_frames: ', task.n_frames)

        # Check if task meets length requirements
        if  kwargs['min_len'] <= task.n_frames <= kwargs['max_len']:
            print('between min_len and max_len')

            imgs, instructions, answer, info_dict = generate_trial(env, task,
                                                                   mode='train' if kwargs['train'] else 'val')
            n_and = instructions.count(' and ')
            n_or = instructions.count(' or ')

            print(n_and, n_or)
            
            # Check if task meets joint operator requirements
            if kwargs['min_joint_ops'] <= (n_and + n_or) <= kwargs['max_joint_ops']:
                print('under bool op limit')
                
                n_delay = task.n_frames - instructions.count('observe')

                # Check if task meets delay frame requirements
                if kwargs['min_delay'] <= n_delay <= kwargs['max_delay']:
                    print('under delay limit')

                    # Check if task features are in balance
                    if kwargs['force_balance']:
                        if (track_tf[answer] + 1) / kwargs['n_trials'] <= 1 / len(track_tf):
                            print('balanced')

                            # Check if task is a duplicate
                            if not duplicate_check(task_ins, instructions):
                                print('not duplicate')

                                track_tf[answer] += 1
                                total_and += n_and
                                total_or += n_or
                                total_not += instructions.count(' not ')
                                task_ins.append(instructions)

                                store_task(task, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')

                                info_dicts = []
                                t_per_t = kwargs['n_trials']//kwargs['n_tasks']
                                i = t_per_t 

                                while i > 0:
                                    imgs, _, _, info_dict = generate_trial(env, task, mode='train' if kwargs['train'] else 'val')
                                    if info_dict not in info_dicts:
                                        read_write.write_trial(imgs, info_dict, os.path.join(kwargs['trials_dir'], 'trial' + 
                                                                                             str(len(tasks) * t_per_t + t_per_t - i)))
                                        info_dicts.append(info_dict)
                                        i -= 1
                                tasks.append(task)
                    else:
                        # Check if task is a duplicate
                        if not duplicate_check(task_ins, instructions):
                            track_tf[answer] += 1
                            total_and += n_and
                            total_or += n_or
                            total_not += instructions.count(' not ')
                            task_ins.append(instructions)

                            store_task(task, kwargs['tasks_dir'] + '/' + str(len(tasks)) + '.json')

                            info_dicts = []
                            t_per_t = kwargs['n_trials'] // kwargs['n_tasks']
                            i = t_per_t

                            while i > 0:
                                imgs, _, _, info_dict = generate_trial(env, task, mode='train' if kwargs['train'] else 'val')

                                if info_dict not in info_dicts:
                                    read_write.write_trial(imgs, info_dict, os.path.join(kwargs['trials_dir'], 'trial' + 
                                                                                         str(len(tasks) * t_per_t + t_per_t - i)))
                                    info_dicts.append(info_dict)
                                    i -= 1

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
    parser.add_argument('--stim_dir', type=str, default='../data/shapenet_handpicked')
    parser.add_argument('--tasks_dir', type=str, default='temp/high_tasks_all_nn_training')
    parser.add_argument('--trials_dir', type=str, default='temp/high_trials_all_nn_training')
    parser.add_argument('--config_path', type=str, default='configs/high_complexity_all.json')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=9)
    parser.add_argument('--min_delay', type=int, default=0)
    parser.add_argument('--max_delay', type=int, default=-1)
    parser.add_argument('--delay_prob', type=float, default=0.5)
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--min_joint_ops', type=int, default=1)
    parser.add_argument('--max_joint_ops', type=int, default=2)
    parser.add_argument('--force_balance', action='store_true', default=False)
    parser.add_argument('--non_bool_actions', action='store_true', default=False)
    args = parser.parse_args()

    print(args)

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
        args.max_delay = 2

    # Initialize environment
    env_spec = env.init_env_spec(
        max_delay=args.max_delay,
        delay_prob=args.delay_prob,
        add_fixation_cue=True,
        auto_gen_config=config,
    )

    env.set_env_spec(env_spec)

    """
    Set balance tracking dictionary 
    (change dictionaries to match feature labels of stimulus set)
    """
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

    print('n_trials:', args.n_trials)
    print('n_tasks:', args.n_tasks)

    print('total:', len(create_tasks(env, track_tf, **vars(args))[0]))

    if args.non_bool_actions:
        number_of_files_to_delete = args.n_trials - n_trials
        delete_last_n_files(args.tasks_dir, number_of_files_to_delete)
        delete_last_n_files(args.trials_dir, number_of_files_to_delete)