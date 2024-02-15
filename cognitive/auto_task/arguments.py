import argparse


def get_args():
    parser = argparse.ArgumentParser(description='auto_task')
    parser.add_argument('--stim_dir', default='../../data/shapenet_handpicked_val',
                        help='directory of stimuli to sample from')
    parser.add_argument('--max_depth', type=int, default=3, help='the depth of the task graph')
    parser.add_argument('--max_op', type=int, default=5, help='the maximum number of operators in the task')
    parser.add_argument('--max_switch', type=int, default=1,
                        help='the maximum number of switch operators in the task graph')
    parser.add_argument('--output_dir', default='./tasks', help='the output directory')
    parser.add_argument(
        '--config_json',
        default='',
        help='Filepath of operator user specified dictionary in json format, '
             'this includes which operators can follow a parent operator in task graphs'
             'as well as the children operator sampling distribution.'
             'users can also specify which operators can be sampled as the root operators, '
             'as well as the set of boolean operators'
             'If not specified, use the default config in auto_task_util'
    )
    parser.add_argument('--n_tasks', type=int, default=1000, help='number of tasks to generate')
    parser.add_argument('--select_limit', action='store_false', default=True)
    parser.add_argument('--switch_threshold', type=float, default=0.3)

    args = parser.parse_args()
    return args
