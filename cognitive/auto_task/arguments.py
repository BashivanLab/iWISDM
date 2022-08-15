import argparse


def get_args():
    parser = argparse.ArgumentParser(description='auto_task')
    parser.add_argument('--stim_dir', default='../../data/MULTIF_5_stim')
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--max_op', type=int, default=10)
    parser.add_argument('--max_switch', type=int, default=2)
    parser.add_argument('--output_dir', default='./graphs')
    parser.add_argument('--operators', nargs='*', default=[])
    parser.add_argument('--n_tasks', type=int, default=100)
    parser.add_argument('--select_limit', action='store_false', default=True)
    parser.add_argument('--switch_threshold', type=float, default=0.3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--fixation_cue', action='store_false', default=True)
    # TODO: add number of trials per task
    args = parser.parse_args()
    return args
