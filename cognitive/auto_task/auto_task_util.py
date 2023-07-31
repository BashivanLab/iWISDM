import json
import shutil
import timeit
from collections import defaultdict
import traceback

from cognitive.auto_task.arguments import get_args
from cognitive import task_generator as tg
from cognitive import constants as const
from cognitive import stim_generator as sg
from cognitive import info_generator as ig

import numpy as np
import random
import networkx as nx
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from typing import Tuple, Union

# Tuples of the graph object, the root root_op number, and the number of operators
GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]
TASK = Tuple[Union[tg.Operator, sg.Attribute], tg.TemporalTask]

# TODO: add select attributes, combine helper with task_generator.task_generation
# root_ops are the operators to begin a task
root_ops = ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "Exist", "IsSame", "And"]
boolean_ops = ["Exist", "IsSame", "And"]
boolean_ops.remove('Exist')
root_ops.remove('Exist')
# uncomment to add ops
# root_ops += ["NotSame", "Or", "Xor"]
# boolean_ops += ["NotSame", "Or", "Xor"]

# all tasks end with select
leaf_op = ["Select"]
mid_op = ["Switch"]

# dictionary specifying which operators can follow an operator,
# e.g. GetCategory follows selecting an object
# 4 operators are related to Select: category, location, view_angle, and the exact object
# if the operator is None, then a random constant is sampled for that attribute
op_dict = {"Select":
               {"n_downstream": 4,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "None"],
                "same_children_op": False
                },
           "GetCategory":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetLoc":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetViewAngle":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetObject":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "Exist":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "IsSame":
               {"n_downstream": 2,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject"],
                "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                "same_children_op": True  # same downstream op
                },
           "And":
               {"n_downstream": 2,
                "downstream": ["Exist", "IsSame", "And"],
                "sample_dist": [1, 0, 0],
                "same_children_op": False
                },
           # "Or":
           #     {"n_downstream": 2,
           #      "downstream": ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
           #      "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
           #      "same_children_op": False
           #      },
           # "Xor":
           #     {"n_downstream": 2,
           #      "downstream": ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
           #      "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
           #      "same_children_op": False
           #      },
           # "NotSame":
           #     {"n_downstream": 2,
           #      "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject"],
           #      "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
           #      "same_children_op": True,
           #      },
           }
op_dict = {"Select":
               {"n_downstream": 4,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "None"],
                "same_children_op": False
                },
           "GetCategory":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetLoc":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetViewAngle":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "GetObject":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]
                },
           "IsSame":
               {"n_downstream": 2,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "CONST"],
                "sample_dist": [0, 0, 0, 0, 1],
                "same_children_op": True  # same downstream op
                },
           "And":
               {"n_downstream": 2,
                "downstream": ["IsSame", "And"],
                "sample_dist": [1, 0],
                "same_children_op": False
                },
           # "Or":
           #     {"n_downstream": 2,
           #      "downstream": ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
           #      "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
           #      "same_children_op": False
           #      },
           # "Xor":
           #     {"n_downstream": 2,
           #      "downstream": ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
           #      "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
           #      "same_children_op": False
           #      },
           # "NotSame":
           #     {"n_downstream": 2,
           #      "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject"],
           #      "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
           #      "same_children_op": True,
           #      },
           }
op_dict = defaultdict(dict, **op_dict)


# uncomment to add more ops
# for op in ['And', 'Or', 'Xor']:
#     op_dict[op]['downstream'] = ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
#     op_dict[op]['sample_dist'] = "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0]


# def switch_generator(conditional, do_if, do_else):
#     nl1, cm1 = conditional
#     nl2, cm2 = do_if
#     nl3, cm3 = do_else
#     l1 = len(nl1)
#     l2 = len(nl2)
#     l3 = len(nl3)
#     node_list = nl1 + nl2 + ["switch"] + nl3
#     conn_mtx = np.zeros((l1 + l2 + l3 + 1, l1 + l2 + l3 + 1))
#     conn_mtx[:l1, :l1] = cm1
#     conn_mtx[l1:l1 + l2, l1:l1 + l2] = cm2
#     conn_mtx[l1 - 1, l1 + l2] = 1
#     conn_mtx[l1 + l2 - 1, l1 + l2] = 1
#     conn_mtx[l1 + l2 + 1:, l1 + l2 + 1:] = cm3
#     conn_mtx[l1 + l2, l1 + l2 + 1] = 1
#     return [node_list, conn_mtx]
#
#
# def subTask_Generator(max_op=32, ):
#     conn_mtx = np.zeros((3 ** max_op, 3 ** max_op))
#     node_list = []
#     root_node = random.choice(root_ops)
#     node_list.append(root_node)
#     curr_node = root_node
#     curr_node_idx = 0
#     pos_node_idx = 0
#     done = False
#
#     while not done:  ## how to constraint the number of operators?
#         curr_node = node_list[curr_node_idx]
#         if op_dict[curr_node]["n_downstream"] == 1:
#             if pos_node_idx > max_op:  # select operators based on sample_dist to limit the depth of the tree
#                 curr_node = random.choices(op_dict[curr_node]["downstream"], op_dict[curr_node]["sample_dist"])[0]
#             else:
#                 curr_node = random.choice(op_dict[curr_node]["downstream"])
#             node_list.append(curr_node)
#             conn_mtx[curr_node_idx, pos_node_idx + 1] = 1
#             curr_node_idx += 1
#             pos_node_idx += 1
#         elif op_dict[curr_node]["n_downstream"] == 2:
#             if pos_node_idx > max_op:
#                 curr_node = random.choices(op_dict[curr_node]["downstream"], op_dict[curr_node]["sample_dist"])[0]
#             else:
#                 curr_node = random.choice(op_dict[curr_node]["downstream"])
#             node_list.append(curr_node)
#             node_list.append(curr_node)
#             conn_mtx[curr_node_idx, pos_node_idx + 1] = 1
#             conn_mtx[curr_node_idx, pos_node_idx + 2] = 1
#             curr_node_idx += 1
#             pos_node_idx += 2
#         elif op_dict[curr_node]["n_downstream"] == 0:
#             if all([op == "Select" for op in node_list[curr_node_idx:pos_node_idx + 1]]):
#                 done = True
#             else:
#                 curr_node_idx += 1
#
#     conn_mtx = conn_mtx[:pos_node_idx + 1, :pos_node_idx + 1]
#     return node_list, conn_mtx

def sample_children_helper(op_name, op_count, max_op, depth, max_depth):
    """
    helper function to ensure the task graph is not too complex
    :param op_name: the current operator
    :param op_count: the current number of operators
    :param max_op: the maximum number of operators allowed
    :param depth: the current depth
    :param max_depth: the maximum depth of the task graph
    :return: a random operator to follow the
    """
    if depth + 1 > max_depth or op_count + 4 > max_op:
        return np.random.choice(op_dict[op_name]["downstream"], p=op_dict[op_name]["sample_dist"])
    else:
        return np.random.choice(op_dict[op_name]["downstream"])


def sample_children_op(op_name, op_count, max_op, depth, max_depth, select_op, select_downstream):
    # bug op_dict is sometimes {}
    n_downstream = op_dict[op_name]["n_downstream"]

    if n_downstream == 1:
        return [random.choice(op_dict[op_name]["downstream"])]
    elif op_name == 'Select':
        ops = list()

        if select_downstream is None:
            select_downstream = op_dict['Select']['downstream']

        if select_op:  # if select at least one operator for select attribute
            get = random.choice(["GetCategory", "GetLoc", "GetViewAngle", "GetObject"])
            ops.append(get)

            if get in select_downstream:
                select_downstream.remove(get)
            n_downstream -= 1
        elif depth + 1 > max_depth or op_count + 1 > max_op:
            return ['None' for _ in range(n_downstream)]

        for _ in range(n_downstream):
            if np.random.random() < 0.8:
                ops.append('None')
            else:
                if select_downstream:
                    ops.append(select_downstream.pop(random.randrange(len(select_downstream))))
                else:
                    ops.append('None')
        return ops
    # TODO: elif op_name == "And", check depth + 2, + 3?
    else:
        if op_dict[op_name]["same_children_op"]:
            child = sample_children_helper(op_name, op_count, max_op, depth, max_depth)
            return [child for _ in range(n_downstream)]
        else:
            return [sample_children_helper(op_name, op_count, max_op, depth, max_depth) for _ in range(n_downstream)]


def branch_generator(G: nx.DiGraph,
                     root_op: str,
                     local_count: int,
                     op_count: int,
                     max_op: int,
                     depth: int,
                     max_depth: int,
                     select_op=False,
                     select_downstream=None) -> int:
    # function to complete a branch of subtask graph based on a root operator,
    # modifies G in place, and returns the leaf node number
    if root_op == 'None':
        return None
    elif root_op == 'CONST':
        return op_count
    else:
        select_op = True if root_op == 'Exist' else select_op
        children = sample_children_op(root_op, local_count, max_op, depth, max_depth, select_op, select_downstream)
        if root_op == 'Select' and any('Get' in c for c in children):
            select_downstream = ['None'] * 4
            select_op = False

        depth += 1
        parent = op_count
        if all(op == 'CONST' for op in children):
            downstream = op_dict['IsSame']['downstream'].copy()
            downstream.remove('CONST')
            children[0] = random.choice(downstream)
        for op in children:
            if op != 'None':
                child = op_count + 1
                local_count += 1
                op_count = branch_generator(G, op, local_count, op_count + 1, max_op, depth, max_depth, select_op,
                                            select_downstream)
                G.add_node(child, label=op)
                G.add_edge(parent, child)
        return op_count


def subtask_graph_generator(count=0, max_op=20, max_depth=10, select_limit=False, root_op=None) -> \
        GRAPH_TUPLE:
    """
    main function for generating subtasks
    uses networkx to compose the task graphs
    :param count: the root_op number of the root
    :param select_limit: whether to add operator after selects. if True, then constants are sampled
    :param root_op: the root operator
    :return: Tuple
    """
    # initialize the graph and save the root_op number of the root
    G = nx.DiGraph()
    root = count

    op_count = count
    root_op = root_op if root_op else random.choice(root_ops)
    G.add_node(op_count, label=root_op)
    select_downstream = ['None'] * 4 if select_limit else None
    local_count = 1
    op_count = branch_generator(G, root_op, local_count, op_count, max_op, 1, max_depth,
                                select_downstream=select_downstream)
    return G, root, op_count


def switch_generator(conditional: GRAPH_TUPLE, do_if, do_else: GRAPH_TUPLE) -> GRAPH_TUPLE:
    # combine the 3 subtasks into the switch task graph by using networkx compose_all
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
    # draw the task graph for visualization
    A = nx.nx_agraph.to_agraph(G)
    A.draw(os.path.join(write_fp, "operator_graph.png"), prog="dot")

    node_labels = {node[0]: node[1]['label'] for node in G.nodes(data=True)}

    # save root_op label dictionary
    with open(os.path.join(write_fp, 'node_labels'), 'w') as f:
        json.dump(node_labels, f, indent=4)
    # save adjacency matrix for reconstruction, use nx.from_dict_of_dicts to reconstruct
    with open(os.path.join(write_fp, 'adj_dict'), 'w') as f:
        json.dump(nx.to_dict_of_dicts(G), f, indent=4)
    task[1].to_json(os.path.join(write_fp, 'temporal_task.json'))
    return None


def write_trial_instance(task: tg.TemporalTask, write_fp: str, img_size=224, fixation_cue=True) -> None:
    # save the actual generated frames into another folder
    if os.path.exists(write_fp):
        shutil.rmtree(write_fp)
    os.makedirs(write_fp)
    frame_info = ig.FrameInfo(task, task.generate_objset())
    compo_info = ig.TaskInfoCompo(task, frame_info)
    objset = compo_info.frame_info.objset
    print(objset)
    for i, (epoch, frame) in enumerate(zip(sg.render(objset, img_size), compo_info.frame_info)):
        # add cross in the center of the image
        if fixation_cue:
            if not any('ending' in description for description in frame.description):
                sg.add_fixation_cue(epoch)
        img = Image.fromarray(epoch, 'RGB') # just store rgb
        filename = os.path.join(write_fp, f'epoch{i}.png')
        img.save(filename)
    _, compo_example, _ = compo_info.get_examples()
    filename = os.path.join(write_fp, 'trial_info')
    with open(filename, 'w') as f:
        json.dump(compo_example, f, indent=4)
    return


if __name__ == '__main__':
    args = get_args()
    print(args)

    const.DATA = const.Data(dir_path=args.stim_dir)

    task_dir = args.task_dir

    if task_dir:
        if not os.path.isdir(task_dir):
            raise ValueError('Task Directory not found')
        start = timeit.default_timer()
        task_folders = [f.path for f in os.scandir(task_dir) if f.is_dir()]
        for f in task_folders:
            try:
                # uncomment to reconstruct the graph
                # labels, adj = os.path.join(f, 'node_labels'), os.path.join(f, 'adj_dict')
                # with open(labels, 'rb') as h:
                #     labels = json.load(h)
                # with open(adj, 'rb') as h:
                #     adj = json.load(h)
                # g = nx.from_dict_of_dicts(adj, create_using=nx.DiGraph)
                # g = nx.relabel_nodes(g, labels)
                # print(sorted(g))
                task_json_fp = os.path.join(f, 'temporal_task.json')
                with open(task_json_fp, 'rb') as h:
                    task_dict = json.load(h)
                task_dict['operator'] = tg.load_operator_json(task_dict['operator'])
                temporal_task = tg.TemporalTask(
                    operator=task_dict['operator'],
                    n_frames=task_dict['n_frames'],
                    first_shareable=task_dict['first_shareable'],
                    whens=task_dict['whens']
                )
                for i in range(args.n_trials):
                    instance_fp = os.path.join(f, f'trial_{i}')
                    if os.path.exists(instance_fp):
                        shutil.rmtree(instance_fp)
                    os.makedirs(instance_fp)

                    write_trial_instance(temporal_task, instance_fp, args.img_size, args.fixation_cue)
            except Exception as e:
                traceback.print_exc()
        stop = timeit.default_timer()
        print('Time taken to generate trials: ', stop - start)
    else:
        start = timeit.default_timer()
        # TODO: check for duplicated tasks by comparing task graphs
        for i in range(args.n_tasks):
            # make directory for saving task information
            fp = os.path.join(args.output_dir, str(i))
            if os.path.exists(fp):
                shutil.rmtree(fp)
            os.makedirs(fp)

            count = 0
            # generate a subtask graph and actual task
            subtask_graph = subtask_graph_generator(count=count, max_op=args.max_op, max_depth=args.max_depth,
                                                    select_limit=args.select_limit)
            subtask = tg.subtask_generation(subtask_graph)
            count = subtask_graph[2] + 1
            for _ in range(args.max_switch):
                if random.random() < args.switch_threshold:  # if add switch
                    new_task_graph = subtask_graph_generator(count=count, max_op=args.max_op, max_depth=args.max_depth,
                                                             select_limit=args.select_limit)
                    count = new_task_graph[2] + 1

                    conditional = subtask_graph_generator(count=count, max_op=args.max_op, max_depth=args.max_depth,
                                                          select_limit=args.select_limit,
                                                          root_op=random.choice(boolean_ops))
                    conditional_task = tg.subtask_generation(conditional)
                    count = conditional[2] + 1
                    if random.random() < 0.5:
                        do_if = subtask_graph
                        do_if_task = subtask
                        do_else = new_task_graph
                        do_else_task = tg.subtask_generation(do_else)
                    else:
                        do_if = new_task_graph
                        do_if_task = tg.subtask_generation(do_if)
                        do_else = subtask_graph
                        do_else_task = subtask
                    subtask_graph = switch_generator(conditional, do_if, do_else)
                    count = subtask_graph[2] + 1
                    subtask = tg.switch_generation(conditional_task, do_if_task, do_else_task)
            # TODO: some guess objset error where ValueError occurs
            # write_instance(subtask_graph, subtask, fp, args.img_size, args.n_trials)
            write_task_instance(subtask_graph, subtask, fp)
        stop = timeit.default_timer()
        print('Time taken to generate tasks: ', stop - start)
