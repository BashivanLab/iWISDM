import argparse


def get_args():
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--max_memory', type=int, default=12)
    parser.add_argument('--max_distractors', type=int, default=0)
    parser.add_argument('--examples_per_family', type=int, default=100)
    parser.add_argument('--output_dir', default='./data')
    parser.add_argument('--stim_dir', default='./data/min_shapenet_easy_angle/')
    parser.add_argument('--random_families', action='store_false', default=True)
    parser.add_argument('--families', nargs='*', default='')
    parser.add_argument('--composition', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--training', type=float, default=0.7)
    parser.add_argument('--validation', type=float, default=0.3)
    parser.add_argument('--nback', type=int, default=-1)
    parser.add_argument('--nback_length', type=int, default=20)
    parser.add_argument('--fix_delay', action='store_true', default=False)
    parser.add_argument('--fixation_cue', action='store_false', default=True)

    args = parser.parse_args()
    return args
