import argparse


def get_args():
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--max_memory', type=int, default=12)
    parser.add_argument('--trials_per_family', type=int, default=10)
    parser.add_argument('--output_dir', default='./data')
    parser.add_argument('--stim_dir', default='./data/MULTIF_5_stim')
    parser.add_argument('--random_families', action='store_false', default=True)
    parser.add_argument('--families', nargs='*', default=['CompareViewAngle'])
    parser.add_argument('--composition', type=int, default=1)
    parser.add_argument('--temporal_switch', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--training', type=float, default=0.7, help='training split')
    parser.add_argument('--validation', type=float, default=0.3, help='validation split')
    parser.add_argument('--nback', type=int, default=0, help='match stimuli n frames earlier in the sequence')
    parser.add_argument('--nback_length', type=int, default=20, help='total length of the nback task')
    parser.add_argument('--seq_length', type=int, default=0)
    parser.add_argument('--seq_reverse', action='store_true', default=False)
    parser.add_argument('--fix_delay', action='store_true', default=False)
    parser.add_argument('--fixation_cue', action='store_false', default=True)
    parser.add_argument('--whens', nargs='*', default=[None])
    parser.add_argument('--first_shareable', type=int, default=None)
    args = parser.parse_args()
    return args
