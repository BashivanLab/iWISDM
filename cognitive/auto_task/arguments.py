import argparse


def get_args():
    parser = argparse.ArgumentParser(description='auto_task')
    parser.add_argument('--max_depth', type=int, default=30)
    parser.add_argument('--max_switch', type=int, default=2)
    parser.add_argument('--subtask_max_depth', type=int, default=12)
    parser.add_argument('--output_dir', default='./graphs')
    parser.add_argument('--operators', nargs='*', default=[])
    parser.add_argument('--composition', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--training', type=float, default=0.7)
    parser.add_argument('--validation', type=float, default=0.3)
    parser.add_argument('--nback', type=int, default=0)
    parser.add_argument('--nback_length', type=int, default=20)
    parser.add_argument('--seq_length', type=int, default=0)
    parser.add_argument('--seq_reverse', action='store_true', default=False)
    parser.add_argument('--fix_delay', action='store_true', default=False)
    parser.add_argument('--fixation_cue', action='store_false', default=True)
    # TODO: examples of nback, and sequential
    args = parser.parse_args()
    return args
