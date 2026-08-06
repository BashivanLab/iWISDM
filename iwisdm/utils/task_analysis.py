"""summary statistics for folders of saved task jsons

Two entry points, both taking a directory of task jsons (as written by
read_write.write_task) and the environment to load them into:

* summarize_tasks: static properties of the tasks themselves
* summarize_trials: properties of the trials those tasks generate, swept over
  the composition hyperparameters given by the caller

Both return (per_item_df, summary_df). per_item_df has one row per task/trial,
summary_df is long-format, with one row per (group, metric, stat, value), so
numeric metrics (mean/std/min/max), value fractions of categorical metrics and
item counts live in one table. The group is the subdirectory for tasks and the
hyperparameter setting for trials; pivot on it for a wide view, e.g.
task_summary.pivot(index=['metric', 'stat'], columns='group', values='value')

Usage:
    env = make(env_id='ShapeNet', dataset_fp='data/shapenet_handpicked')
    per_task, task_summary = summarize_tasks(env, 'datasets/no_const_fixed')
    per_trial, trial_summary = summarize_trials(
        env, 'datasets/no_const_fixed', n_trials=50, num_merge=[0, 1, 2, 4])
"""

import os
import random
from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import iwisdm.envs.shapenet.registration as env_reg
import iwisdm.envs.shapenet.task_generator as tg
from iwisdm.envs.shapenet.shapenet_env import ShapeNetEnv

# per-item columns that are identifiers or free text rather than metrics
NON_METRIC_COLS = {'path', 'instruction', 'error'}


class LoadedTask(NamedTuple):
    """one task json, either loaded (task is set) or failed (error is set)"""
    group: str
    path: str
    task: Optional[tg.TemporalTask]
    error: str


def load_tasks(
        env: ShapeNetEnv,
        task_dir: str,
        max_tasks_per_group: int = None,
        seed: int = 0,
) -> List[LoadedTask]:
    """
    Load every task json under task_dir with env.read_task

    Subdirectories are the groups the task summary breaks down by; the dataset
    folders use one subdirectory per frame count (datasets/no_const_fixed/3f ...
    10f). Jsons directly in task_dir form the '.' group.

    @param env: the environment the tasks are loaded into
    @param task_dir: directory of task jsons, searched recursively
    @param max_tasks_per_group: if set, sample at most this many jsons per group.
    Loading is ~6ms per task, so the full datasets/no_const_fixed (~74k jsons)
    takes several minutes
    @param seed: seed for the sampling, and for the global rngs (see _seed_rngs)
    @return: one LoadedTask per json, in group and file order
    """
    _seed_rngs(seed)
    sampler = random.Random(seed)

    loaded = list()
    for group, files in sorted(_group_files(task_dir).items()):
        if max_tasks_per_group and len(files) > max_tasks_per_group:
            files = sorted(sampler.sample(files, max_tasks_per_group))
        for fp in files:
            try:
                loaded.append(LoadedTask(group, fp, env.read_task(fp), ''))
            except Exception as e:
                loaded.append(LoadedTask(group, fp, None, _err(e)))
    return loaded


def summarize_tasks(
        env: ShapeNetEnv,
        task_dir: str,
        max_tasks_per_group: int = None,
        seed: int = 0,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize the tasks saved in task_dir, broken down by subdirectory

    Each task is described by its graph (frame count, shareability, operator
    counts, attributes it attends to, attribute comparisons it makes) and by one
    instantiation of it: the task is given an objset, which fixes its stimuli,
    delay frames and answer.

    @param env: the environment the tasks are loaded into
    @param task_dir: directory of task jsons, searched recursively
    @param max_tasks_per_group: if set, sample at most this many jsons per subdirectory
    @param seed: seed for the sampling and for the global rngs (see _seed_rngs)
    @param verbose: print the summary table
    @return: (one row per task, long-format summary per subdirectory)
    """
    loaded = load_tasks(env, task_dir, max_tasks_per_group, seed)

    per_task = pd.DataFrame([_task_row(lt) for lt in loaded])
    # a task that never uses an operator has no count for it, not a missing one.
    # which operators appear depends on the tasks, so order them for a stable summary
    op_cols = sorted(c for c in per_task.columns if c.startswith('n_op_'))
    ok = per_task['error'] == ''
    per_task.loc[ok, op_cols] = per_task.loc[ok, op_cols].fillna(0)
    per_task = per_task[[c for c in per_task.columns if c not in op_cols] + op_cols]

    summary = _summarize(per_task, ['group'], categorical=['root_op', 'answer'])
    summary = _add_comparison_shares(summary, ['group'])
    if verbose:
        _print_summary(summary, ['group'], f'{len(per_task)} tasks from {task_dir}')
    return per_task, summary


def summarize_trials(
        env: ShapeNetEnv,
        task_dir: str,
        n_trials: int = 20,
        num_merge: Union[int, Sequence[int]] = (0, 1, 2),
        add_distractor_frame: Union[int, Sequence[int]] = 0,
        add_distractor_time: Union[int, Sequence[int]] = 0,
        mode: Union[str, Sequence[str]] = None,
        max_tasks_per_group: int = None,
        seed: int = 0,
        verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize the trials the tasks in task_dir generate, per hyperparameter setting

    All tasks under task_dir form one merge pool, regardless of subdirectory, so
    tasks of different lengths can be composed into the same trial. Each trial
    picks a random base task from the pool and merges num_merge randomly picked
    tasks into it (env.merge_tasks), then renders it (TaskInfoCompo.generate_trial,
    via env.generate_trials). Merging can fail when a task's stimuli cannot be
    reused; those trials are kept as rows with an error and reported as an
    error_rate per setting.

    The sweep is the product of the hyperparameters, each of which takes a single
    value or a sequence of them, with n_trials trials per combination. Rendering
    dominates the runtime at roughly 0.1s per trial.

    @param env: the environment the tasks are loaded into
    @param task_dir: directory of task jsons, searched recursively
    @param n_trials: trials to generate per hyperparameter combination
    @param num_merge: how many tasks to merge into the base task
    @param add_distractor_frame: distractors to add to frames that hold a stimulus
    @param add_distractor_time: distractors to add to empty (delay) frames
    @param mode: stimuli split to render from, e.g. 'train', 'val'; None uses all stimuli
    @param max_tasks_per_group: if set, sample at most this many jsons per
    subdirectory into the merge pool
    @param seed: seed for the sampling and for the global rngs (see _seed_rngs)
    @param verbose: print the summary table
    @return: (one row per trial, long-format summary per hyperparameter setting)
    """
    loaded = load_tasks(env, task_dir, max_tasks_per_group, seed)
    # the pool is only a source of tasks to pick from: merge_tasks re-instantiates
    # each picked task, so the stimuli of a trial come from the split set below
    pool = env.init_compositional_tasks([lt.task for lt in loaded if lt.task is not None])
    if not pool:
        raise ValueError(f'no tasks could be loaded from {task_dir}')

    setting_cols = ['num_merge', 'add_distractor_frame', 'add_distractor_time', 'mode']
    rows = list()
    for setting in product(
            _as_tuple(num_merge),
            _as_tuple(add_distractor_frame),
            _as_tuple(add_distractor_time),
            _as_tuple(mode),
    ):
        kwargs = dict(zip(setting_cols, setting))
        for _ in range(n_trials):
            rows.append({**kwargs, **_trial_row(env, pool, **kwargs)})

    per_trial = pd.DataFrame(rows)
    summary = _summarize(per_trial, setting_cols, categorical=['final_answer'])
    summary = _add_comparison_shares(summary, setting_cols)
    if verbose:
        _print_summary(
            summary, setting_cols,
            f'{len(per_trial)} trials from a pool of {len(pool)} tasks in {task_dir}'
        )
    return per_trial, summary


def _seed_rngs(seed: int) -> None:
    # task instantiation, merging and stimulus sampling all draw from the global
    # rngs, so summaries are only reproducible if both of them are seeded
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _err(e: Exception) -> str:
    return f'{type(e).__name__}: {e}'


def _as_tuple(value) -> Tuple:
    # hyperparameters are given either as a single value or a sequence of them
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _group_files(task_dir: str) -> Dict[str, List[str]]:
    # map each subdirectory of task_dir to the task jsons it holds
    groups = defaultdict(list)
    for dir_path, _, file_names in os.walk(task_dir):
        group = os.path.relpath(dir_path, task_dir)
        for fn in file_names:
            if fn.endswith('.json'):
                groups[group].append(os.path.join(dir_path, fn))
    if not groups:
        raise ValueError(f'no task jsons found in {task_dir}')
    return groups


def _comparison_counts(nodes: List) -> Counter:
    """
    Count the attribute comparisons a task graph makes, keyed by attribute

    A comparison is an IsSame or NotSame operator reading its operands off
    stimuli, i.e. one applied to Get operators (GetCategory, GetObject, GetLoc,
    GetViewAngle). Both operands of a Same always carry the same attr_type, so
    each Same counts once, towards the attribute it compares.

    @param nodes: the task graph's nodes, from task.topological_sort()
    @return: attribute name -> number of comparisons of that attribute
    """
    counts = Counter()
    for node in nodes:
        if isinstance(node, tg.Same) and any(isinstance(c, tg.Get) for c in node.child):
            counts[node.attr_type] += 1
    return counts


def _task_stats(task: tg.TemporalTask) -> Dict:
    # instantiating the task fixes its stimuli, its delay frames and its answer
    objset = task.generate_objset()
    nodes = task.topological_sort()
    ops = Counter(type(node).__name__ for node in nodes)
    attrs = set().union(*[task.get_relevant_attribute(f'last{k}') for k in range(task.n_frames)])
    comparisons = _comparison_counts(nodes)
    instruction = str(task)

    stats = {
        'n_frames': task.n_frames,
        'first_shareable': task.first_shareable,
        'shareable': int(task.first_shareable < task.n_frames),
        'operator_size': task.operator_size,
        'n_comparisons': sum(comparisons.values()),
        'root_op': type(nodes[0]).__name__,
        'n_objects': len(objset),
        'n_delay_frames': sum(1 for i in range(task.n_frames) if not objset.dict.get(i)),
        'answer': str(env_reg.get_target_value(task.get_target(objset)[0])),
        'instruction': instruction,
        'instruction_words': len(instruction.split()),
    }
    stats.update({f'n_op_{name}': count for name, count in ops.items()})
    stats.update({f'uses_{attr}': int(attr in attrs) for attr in env_reg.DATA.ATTRS})
    stats.update({f'n_compare_{attr}': comparisons[attr] for attr in env_reg.DATA.ATTRS})
    return stats


def _task_row(loaded: LoadedTask) -> Dict:
    row = {'group': loaded.group, 'path': loaded.path, 'error': loaded.error}
    if loaded.task is None:
        return row
    try:
        row.update(_task_stats(loaded.task))
    except Exception as e:
        row['error'] = _err(e)
    return row


def _trial_row(
        env: ShapeNetEnv,
        pool: List,
        num_merge: int,
        add_distractor_frame: int,
        add_distractor_time: int,
        mode: str,
) -> Dict:
    try:
        # merging instantiates stimuli, so the split has to be set before it, not
        # only inside generate_trials
        env.reset_env(mode)
        compo = env.merge_tasks(compositional_infos=pool, num_merge=num_merge)
        _, _, compo_info = env.generate_trials(
            compositional_infos=[compo],
            mode=mode,
            add_distractor_frame=add_distractor_frame,
            add_distractor_time=add_distractor_time,
        )[0]
    except Exception as e:
        return {'error': _err(e)}

    objs = list(compo.frame_info.objset)
    responses = [a for a in compo_info['answers'] if a != 'null']
    instruction = compo_info['instruction']
    # the comparisons of a trial are those of every task composed into it
    comparisons = Counter()
    for task in compo.tasks:
        comparisons.update(_comparison_counts(task.topological_sort()))
    return {
        'error': '',
        'n_tasks': len(compo.tasks),
        'n_frames': compo_info['epochs'],
        'n_objects': len(objs),
        'n_distractors': sum(1 for o in objs if o.deletable),
        'n_delay_frames': sum(1 for frame in compo.frame_info if not frame.objs),
        'n_response_frames': len(responses),
        'n_true': responses.count('true'),
        'n_false': responses.count('false'),
        'n_comparisons': sum(comparisons.values()),
        **{f'n_compare_{attr}': comparisons[attr] for attr in env_reg.DATA.ATTRS},
        'final_answer': responses[-1] if responses else '',
        'instruction': instruction,
        'instruction_words': len(instruction.split()),
    }


def _summarize(
        per_item: pd.DataFrame,
        group_cols: List[str],
        categorical: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Aggregate a per-item frame into long format

    @param per_item: one row per task or trial, with an error column
    @param group_cols: columns to group by, kept as columns of the summary
    @param categorical: columns summarized as the fraction of items per value
    @return: one row per (group columns, metric, stat, value), where stat is one
    of count, mean, std, min, max, frac. Items that errored are counted by the
    error_rate metric and excluded from every other metric
    """
    # a column is missing when every item errored before it could be filled in
    categorical = [c for c in categorical if c in per_item.columns]
    skip = set(group_cols) | set(categorical) | NON_METRIC_COLS
    metrics = [c for c in per_item.columns if c not in skip]
    # every group reports every observed value, so a value it never takes is 0, not missing
    values_seen = {col: sorted(per_item[per_item['error'] == ''][col].unique()) for col in categorical}

    rows = list()
    for keys, group in per_item.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group_id = dict(zip(group_cols, keys))
        ok = group[group['error'] == '']

        rows.append({**group_id, 'metric': 'n', 'stat': 'count', 'value': float(len(group))})
        rows.append({**group_id, 'metric': 'error_rate', 'stat': 'frac',
                     'value': float((group['error'] != '').mean())})
        values = ok[metrics].astype(float)
        for metric in metrics:
            for stat in ('mean', 'std', 'min', 'max'):
                rows.append({**group_id, 'metric': metric, 'stat': stat,
                             'value': getattr(values[metric], stat)()})
        for col in categorical:
            fracs = ok[col].value_counts(normalize=True)
            for value in values_seen[col]:
                rows.append({**group_id, 'metric': f'{col}={value}', 'stat': 'frac',
                             'value': float(fracs.get(value, 0.0))})
    return pd.DataFrame(rows)


def _add_comparison_shares(summary: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Append each attribute's share of a group's attribute comparisons

    The share is pooled over the group's items, i.e. the mean count for one
    attribute over the mean count of all comparisons, so items making many
    comparisons weigh proportionally more than in an average of per-item shares.

    @param summary: long-format summary from _summarize
    @param group_cols: the columns it is grouped by
    @return: the summary with one pct_compare_<attr> metric added per attribute
    """
    counts = summary[(summary['stat'] == 'mean')
                     & summary['metric'].str.startswith('n_compare_')].copy()
    if counts.empty:
        return summary

    totals = counts.groupby(group_cols, dropna=False)['value'].transform('sum')
    counts['value'] = 100 * counts['value'] / totals.replace(0, np.nan)
    counts['metric'] = counts['metric'].str.replace('n_compare_', 'pct_compare_', regex=False)
    counts['stat'] = 'pct'
    return pd.concat([summary, counts], ignore_index=True)


def _print_summary(summary: pd.DataFrame, group_cols: List[str], title: str) -> None:
    # metrics as rows, groups as columns; numeric metrics as mean +- std
    summary = summary.copy()
    summary['group'] = summary[group_cols].astype(str).agg(', '.join, axis=1)

    columns = dict()
    for metric, rows in summary.groupby('metric', sort=False):
        stats = rows.pivot(index='group', columns='stat', values='value')
        if {'mean', 'std'} <= set(stats.columns):
            column = stats.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)
        elif 'count' in stats.columns:
            column = stats['count'].map(lambda v: f'{v:,.0f}')
        elif 'pct' in stats.columns:
            column = stats['pct'].map(lambda v: f'{v:.1f}%')
        else:
            column = stats[stats.columns[0]].map(lambda v: f'{v:.3f}')
        columns[metric] = column

    print(f'\n{title}')
    print(f'columns: {", ".join(group_cols)}')
    print(pd.DataFrame(columns).T.to_string())
