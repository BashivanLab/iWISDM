import argparse


def get_args():
    parser = argparse.ArgumentParser(description='auto_task')
    parser.add_argument('--task_dir', default='', help='directory to pre-generated tasks')
    parser.add_argument('--stim_dir', default='../../data/MULTIF_5_stim', help='directory of stimuli to sample from')
    parser.add_argument('--max_depth', type=int, default=3, help='the depth of the task graph')
    parser.add_argument('--max_op', type=int, default=5, help='the maximum number of operators in the task')
    parser.add_argument('--max_switch', type=int, default=1,
                        help='the maximum number of switch operators in the task graph')
    parser.add_argument('--output_dir', default='./graphs', help='the output directory')
    parser.add_argument('--operators', nargs='*', default=[], help='the allowed operators')
    parser.add_argument('--n_tasks', type=int, default=10000, help='number of tasks to generate')
    parser.add_argument('--n_trials', type=int, default=10, help='number of task instances per random task')
    parser.add_argument('--select_limit', action='store_false', default=True)
    parser.add_argument('--switch_threshold', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--fixation_cue', action='store_false', default=True)
    # TODO: add number of trials per task
    args = parser.parse_args()
    return args
