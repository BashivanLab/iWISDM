import json
import shutil
from collections import defaultdict

from cognitive.auto_task.arguments import get_args
from cognitive import task_generator as tg
from cognitive import stim_generator as sg
from cognitive import info_generator as ig
from cognitive import constants as const

import numpy as np
import random
import timeit
import networkx as nx
from PIL import Image
import os

from typing import Tuple, Union, List

# Tuples of the graph object, the root root_op number, and the number of operators,
# needed to compose graphs in switch generation
GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]
TASK = Tuple[Union[tg.Operator, sg.Attribute], tg.TemporalTask]

# root_ops are the operators to begin a task
root_ops = ["IsSame", "And", "Or", "NotSame", "GetLoc", "GetCategory"]
boolean_ops = ["IsSame", "And", "Or", "NotSame", ]

# all tasks end with select
leaf_op = ["Select"]
mid_op = ["Switch"]

# dictionary specifying which operators can follow an operator,
# e.g. GetCategory follows selecting an object
# 4 operators are related to Select: category, location, view_angle, and the exact object
# if the operator is None, then a random constant is sampled for that attribute
op_dict = {
    "Select":
        {
            "n_downstream": 4,
            "downstream": ["GetLoc", "GetCategory", "GetObject"],
            "same_children_op": False,
            "min_depth": 1,
            "min_op": 1,
        },
    "GetCategory":
        {
            "n_downstream": 1,
            "downstream": ["Select"],
            "min_depth": 2,
            "min_op": 2,
        },
    "GetLoc":
        {
            "n_downstream": 1,
            "downstream": ["Select"],
            "min_depth": 2,
            "min_op": 2,
        },
    "GetObject":
        {
            "n_downstream": 1,
            "downstream": ["Select"],
            "min_depth": 2,
            "min_op": 2,
        },
    "IsSame":
        {
            "n_downstream": 2,
            "downstream": ["GetLoc", "GetCategory", "GetObject", "CONST"],
            "sample_dist": [4 / 15, 4 / 15, 4 / 15, 1 / 5],

            "same_children_op": True,
            "min_depth": 3,
            "min_op": 7,
        },
    "NotSame":
        {
            "n_downstream": 2,
            "downstream": ["GetLoc", "GetCategory", "GetObject", "CONST"],
            "sample_dist": [4 / 15, 4 / 15, 4 / 15, 1 / 5],
            "same_children_op": True,
            "min_depth": 3,
            "min_op": 7,
        },
    "And":
        {
            "n_downstream": 2,
            "downstream": ["IsSame", "NotSame", "And", "Or"],
            "same_children_op": False,
            "min_depth": 4,
            "min_op": 15,
        },
    "Or":
        {
            "n_downstream": 2,
            "downstream": ["IsSame", "NotSame", "And", "Or"],
            "same_children_op": False,
            "min_depth": 4,
            "min_op": 15,
        },
    "CONST":
        {
            "n_downstream": 0,
            "downstream": [],
            "sample_dist": [],
            "same_children_op": False,
            "min_depth": 1,
            "min_op": 1,
        }
}

op_dict = defaultdict(dict, **op_dict)
op_depth_limit = {k: v['min_depth'] for k, v in op_dict.items()}
op_operators_limit = {k: v['min_op'] for k, v in op_dict.items()}


def count_depth_and_op(op):
    # return the operator count and the depth of the operator graph
    op_count, depth_count = 0, 0
    if not isinstance(op, tg.Operator):
        return op_count, depth_count
    if op.child:
        # iterate over the nodes, and increment count
        depth_counts = list()
        for c in op.child:
            add_count = count_depth_and_op(c)
            op_count += add_count[0]
            depth_counts += [add_count[1]]
        depth_count += max(depth_counts)
    else:
        # at the leaf nodes
        return 1, 1
    op_count += 1
    depth_count += 1
    return op_count, depth_count


def sample_root_helper(max_op, max_depth):
    depth_filter = [op for op, v in op_depth_limit.items() if (v <= max_depth) and (op in root_ops)]
    both_filter = [op for op in depth_filter if op_operators_limit[op] <= max_op]
    if not both_filter:
        depth = {op: v for op, v, in op_depth_limit.items() if op in root_ops}
        root_op = {op: v for op, v, in op_operators_limit.items() if op in root_ops}
        raise ValueError(f'The specified task complexity is too low given the available root operators\n'
                         f'The minimum depth of the specified root operators is: {depth}\n'
                         f'The minimum depth of the specified root operators is: {root_op}')
    return np.random.choice(both_filter)


def sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth):
    """
    helper function to ensure the task complexity is bounded, and return the sampled child operator
    max_depth is a tighter bound than max_op since operators have to complete their subtask graph
    :param op_name: the current operator
    :param op_count: the current number of operators
    :param max_op: the maximum number of operators allowed
    :param cur_depth: the current depth
    :param max_depth: the maximum depth of the task graph
    :return: a randomly sampled operator to follow the parent node
    """

    downstream_ops = op_dict[op_name]["downstream"]
    min_add_depth_filter = {op: (max_depth - (cur_depth + op_depth_limit[op] - 1)) for op in downstream_ops if
                            (cur_depth + op_depth_limit[op] <= max_depth)}
    min_add_op_filter = {op: (max_op - (op_count + op_operators_limit[op])) for op in downstream_ops if
                         (op_count + op_operators_limit[op] <= max_op)}

    complexity_downstream = [op_depth_limit[op] for op in downstream_ops]
    min_complexity_downstream = [op for op in downstream_ops if op_depth_limit[op] == min(complexity_downstream)]
    if "sample_dist" in op_dict[op_name]:
        return np.random.choice(op_dict[op_name]["downstream"], p=op_dict[op_name]["sample_dist"])
    if not min_add_depth_filter:
        return np.random.choice(min_complexity_downstream)

    if max(min_add_depth_filter.values()) > 0:
        # if added operator sub-graph can fit and have left over depth
        filtered_ops = [op for op, diff in min_add_depth_filter.items() if diff > 0]
        both_filter = {k: v for k, v in min_add_op_filter.items() if k in filtered_ops}
        if both_filter:
            if max(both_filter.values()) > 0:
                both_filter = [k for k, v in both_filter.items() if v > 0]
                return np.random.choice(both_filter)
            elif max(both_filter.values()) == 0:
                both_filter = [k for k, v in both_filter.items() if v == 0]
                return np.random.choice(both_filter)
        return np.random.choice(min_complexity_downstream)
    elif max(min_add_depth_filter.values()) == 0:
        # if only certain operator sub-graphs can fit
        filtered_ops = [op for op, diff in min_add_depth_filter.items() if diff == 0]
        both_filter = {k: v for k, v in min_add_op_filter.items() if k in filtered_ops}
        if both_filter:
            if max(both_filter.values()) >= 0:
                both_filter = [k for k, v in both_filter.items() if v == max(both_filter.values())]
                return np.random.choice(both_filter)
        # cannot do anything about bounding, just complete the minimum task graph
        return np.random.choice(min_complexity_downstream)
    else:
        # cannot do anything about bounding, just complete the minimum task graph
        return np.random.choice(min_complexity_downstream)


def sample_children_op(
        op_name: str,
        op_count: int,
        max_op: int,
        cur_depth: int,
        max_depth: int,
        select_op: bool,
        select_downstream: List[str]
) -> List[str]:
    """
    sample the children operators given the operator name
    :param op_name: the name of the parent node
    :param op_count: operator count
    :param max_op: max number of operators allowed
    :param cur_depth: current depth
    :param max_depth: max depth allowed for the task graph
    :param select_op: Boolean, does select follow an operator
    :param select_downstream: the downstream options that can be sampled from
    :return: list of operators
    """

    n_downstream = op_dict[op_name]["n_downstream"]  # how many children need to be sampled
    if n_downstream == 1:
        return [sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)]
    elif op_name == 'Select':
        children = list()  # append children operators

        if cur_depth + 2 > max_depth or op_count + 2 > max_op:
            return ['None' for _ in range(n_downstream)]
        else:
            op_sampled = False
            if select_op:  # if Select is followed by a Get operator
                if select_downstream is None:
                    select_downstream = op_dict['Select']['downstream']
                for _ in range(n_downstream):
                    if np.random.random() < 0.8:  # make sure not too many operators follow select
                        children.append('None')
                    else:
                        if select_downstream and not op_sampled:  # if the list is not empty, sample a downstream op
                            get = random.choice(select_downstream)
                            children.append(get)
                            op_sampled = True
                            select_downstream.remove(get)
                        else:
                            children.append('None')
                return children
            else:
                return ['None' for _ in range(n_downstream)]
    else:
        if op_dict[op_name]["same_children_op"]:
            child = sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)
            return [child for _ in range(n_downstream)]
        else:
            return [sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth) for _ in
                    range(n_downstream)]


def branch_generator(
        G: nx.DiGraph,
        root_op: str,
        op_count: int,
        max_op: int,
        cur_depth: int,
        max_depth: int,
        select_op=False,
        select_downstream=None
) -> int:
    # function to complete a branch of subtask graph based on a root operator,
    # modifies G in place, and returns the leaf node number
    if root_op == 'None':
        return None  # don't add any nodes or edges
    elif root_op == 'CONST':
        return op_count  # don't add any nodes or edges
    else:
        # Exist operator always follows a select with operator as child
        select_op = True if root_op == 'Exist' else select_op

        children = sample_children_op(
            op_name=root_op,
            op_count=op_count + 1,
            max_op=max_op,
            cur_depth=cur_depth + 1,
            max_depth=max_depth,
            select_op=select_op,
            select_downstream=select_downstream
        )
        cur_depth += 1  # increment cur_depth count

        if root_op == 'Select' and any('Get' in c for c in children):
            # stop generating overly complex tasks where selects with operator children follow more operators
            select_downstream = ['None'] * 4
            select_op = False

        parent = op_count
        if root_op in ['IsSame', 'Or', 'NotSame']:
            if all(op == 'CONST' for op in children):
                # make sure we are not comparing two constants in IsSame
                downstream = op_dict[root_op]['downstream'].copy()
                downstream.remove('CONST')
                if 'sample_dist' in op_dict[root_op]:
                    downstream = [op for op in downstream
                                  if op_dict[root_op]['sample_dist'][op_dict[root_op]['downstream'].index(op)] > 0.0]
                children[0] = random.choice(downstream)  # add a Get op to compare with the constant
            if any(op == 'CONST' for op in children):
                const_idx = children.index('CONST')
                other_idx = 0 if const_idx == 1 else 1
                # remove this check if there's mapping for object idx
                if any(op == 'GetViewAngle' or op == 'GetObject' for op in children):
                    children[const_idx] = children[other_idx]  # add a Get op to compare with the constant
        for op in children:  # loop over sampled children and modify the graph in place
            if op != 'None':  # if the operator is an operator
                child = op_count + 1
                # recursively generate branches based on the child operator
                # op_count is incremented based on how many nodes were added in the child branch call
                op_count = branch_generator(G, op, child, max_op, cur_depth, max_depth, select_op,
                                            select_downstream)
                G.add_node(child, label=op)  # modify the graph
                G.add_edge(parent, child)
        return op_count


def subtask_graph_generator(count=0, max_op=20, max_depth=10, select_limit=False, root_op=None) -> \
        GRAPH_TUPLE:
    """
    function for generating subtask graphs
    uses networkx to compose the task graphs
    :param count: the root_op number of the root
    :param select_limit: whether to add operator after selects. if True, then constants are sampled.
    If false, then operators could be sampled
    :param root_op: the root operator
    :return: GRAPH_TUPLE of (nx.DiGraph(), node number of the graph root, operator count)
    """
    # initialize the graph and save the root_op number of the root
    G = nx.DiGraph()
    root = count

    op_count = count

    root_op = root_op if root_op else sample_root_helper(max_op, max_depth)
    G.add_node(op_count, label=root_op)

    select_downstream = ['None'] * 4 if select_limit else None

    op_count = branch_generator(G, root_op, op_count, max_op, 0, max_depth,
                                select_downstream=select_downstream)
    return G, root, op_count


def task_generator(
        max_switch: int,
        switch_threshold: float,
        max_op: int,
        max_depth: int,
        select_limit: bool
) -> Tuple[
    GRAPH_TUPLE, TASK]:
    """
    function to generate a random task graph and corresponding task
    :param max_switch: the maximum number of switch operators allowed
    :param switch_threshold: float in [0,1], how likely a switch operator is sampled, higher is more likely
    :param max_op: max number of operators allowed
    :param max_depth: max depth of the task graph
    :param select_limit: whether to add operator after selects. if True, then constants are sampled.
    If false, then operators could be sampled
    :return: The random task graph tuple and task tuple
    """
    count = 0
    whens = dict()

    # generate a subtask graph and the actual task
    subtask_graph = subtask_graph_generator(count=count, max_op=max_op, max_depth=max_depth,
                                            select_limit=select_limit)
    subtask, whens = tg.subtask_generation(subtask_graph, existing_whens=whens)
    count = subtask_graph[2] + 1  # start a new subtask graph node number according to old graph
    for _ in range(max_switch):
        if random.random() < switch_threshold:  # if add switch
            new_task_graph = subtask_graph_generator(count=count, max_op=max_op, max_depth=max_depth,
                                                     select_limit=select_limit)
            count = new_task_graph[2] + 1
            conditional = subtask_graph_generator(count=count, max_op=max_op, max_depth=max_depth,
                                                  select_limit=select_limit,
                                                  root_op=random.choice(boolean_ops))
            conditional_task, whens = tg.subtask_generation(conditional, existing_whens=whens)
            if random.random() < 0.5:  # randomly split the do_if and do_else tasks
                do_if = subtask_graph
                do_if_task = subtask
                do_else = new_task_graph
                do_else_task, whens = tg.subtask_generation(do_else, existing_whens=whens)
            else:
                do_if = new_task_graph
                do_if_task, whens = tg.subtask_generation(do_if, existing_whens=whens)
                do_else = subtask_graph
                do_else_task = subtask
            subtask_graph = switch_generator(conditional, do_if, do_else)
            count = subtask_graph[2] + 1
            subtask = tg.switch_generation(conditional_task, do_if_task, do_else_task, whens)
    return subtask_graph, subtask


def switch_generator(conditional: GRAPH_TUPLE, do_if: GRAPH_TUPLE, do_else: GRAPH_TUPLE) -> GRAPH_TUPLE:
    # combine the 3 subtasks graphs into the switch task graph by using networkx compose_all
    do_if_graph, do_if_root, do_if_node = do_if
    do_else_graph, do_else_root, do_else_node = do_else
    conditional_graph, conditional_root, conditional_node = conditional

    # combine all 3 graphs and add edges
    G: nx.DiGraph = nx.compose_all([do_if_graph, do_else_graph, conditional_graph])
    switch_count = conditional_node + 1
    G.add_node(switch_count, label='Switch')
    # note that all directed edges are reversed later at task construction
    # after reversal, switch operator is connected to the selects of do_if and do_else
    # the root of conditional is then connected to the switch operator
    G.add_edge(do_if_node, switch_count)
    G.add_edge(do_else_node, switch_count)
    G.add_edge(switch_count, conditional_root)
    return G, switch_count, switch_count


def write_task_instance(G_tuple: GRAPH_TUPLE, task: TASK, write_fp: str):
    G, _, _ = G_tuple
    G = G.reverse()
    # uncomment to draw the task graph for visualization, super slow
    # A = nx.nx_agraph.to_agraph(G)
    # A.draw(os.path.join(fp, "operator_graph.png"), prog="dot")

    node_labels = {node[0]: node[1]['label'] for node in G.nodes(data=True)}

    # save root_op label dictionary
    with open(os.path.join(write_fp, 'node_labels'), 'w') as f:
        json.dump(node_labels, f, indent=4)
    # save adjacency matrix for reconstruction, use nx.from_dict_of_dicts to reconstruct
    with open(os.path.join(write_fp, 'adj_dict'), 'w') as f:
        json.dump(nx.to_dict_of_dicts(G), f, indent=4)
    task[1].to_json(os.path.join(write_fp, 'temporal_task.json'))
    return None


def write_trial_instance(
        task: tg.TemporalTask,
        write_fp: str,
        img_size=224,
        fixation_cue=True,
        train=True,
        is_instruction=True,
        external_instruction=None
) -> None:
    # TODO: drawing the frames is slow!
    # save the actual generated frames into another folder
    if os.path.exists(write_fp):
        shutil.rmtree(write_fp)
    os.makedirs(write_fp)
    frame_info = ig.FrameInfo(task, task.generate_objset())
    compo_info = ig.TaskInfoCompo(task, frame_info)
    objset = compo_info.frame_info.objset
    for i, (epoch, frame) in enumerate(zip(sg.render(objset, img_size), compo_info.frame_info)):
        # add cross in the center of the image
        if fixation_cue:
            if not any('ending' in description for description in frame.description):
                sg.add_fixation_cue(epoch)
        img = Image.fromarray(epoch, 'RGB')
        filename = os.path.join(write_fp, f'epoch{i}.png')
        # this is slow!
        img.save(filename)
    _, compo_example = compo_info.get_examples(is_instruction=is_instruction,
                                               external_instruction=external_instruction)  # xlei: orginally have three outputs
    filename = os.path.join(write_fp, 'trial_info')
    with open(filename, 'w') as f:
        json.dump(compo_example, f, indent=4)
    return


if __name__ == '__main__':
    args = get_args()
    print(args)

    const.DATA = const.Data(
        dir_path=args.stim_dir
    )
    if args.op_dict_json:
        with open(args.op_dict_json) as f:
            op_dict = json.load(f)

    start = timeit.default_timer()
    for i in range(args.n_tasks):
        # make directory for saving task information
        fp = os.path.join(args.output_dir, str(i))
        if os.path.exists(fp):
            shutil.rmtree(fp)
        os.makedirs(fp)

        task_graph, task = task_generator(
            max_switch=args.max_switch,
            switch_threshold=args.switch_threshold,
            max_op=args.max_op,
            max_depth=args.max_depth,
            select_limit=args.select_limit
        )
        write_task_instance(task_graph, task, fp)
    stop = timeit.default_timer()
    print('Time taken to generate tasks: ', stop - start)
