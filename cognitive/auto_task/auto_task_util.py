import numpy as np
import random
import networkx as nx
import pickle
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# TODO: add select attributes, combine helper with task_generator.task_generation
root_ops = ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "Exist", "IsSame", "And"]
# root_ops += ["NotSame", "Or", "Xor"]
leaf_op = ["Select"]
mid_op = ["Switch"]

op_dict = {"Select":
               {"n_downstream": 4,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "None"],
                "dist": [3 / 80, 3 / 80, 3 / 80, 3 / 80, 17 / 20],
                "sample_dist": [0, 0, 0, 0, 1],
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
           "NotSame":
               {"n_downstream": 2,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject"],
                "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                "same_children_op": True,
                },
           "And":
               {"n_downstream": 2,
                "downstream": ["Exist", "IsSame", "NotSame", "And", "Or", "Xor"],
                "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
                "same_children_op": False
                },
           "Or":
               {"n_downstream": 2,
                "downstream": ["Exist", "IsSame", "NotIsSame", "And", "Or", "Xor"],
                "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
                "same_children_op": False
                },
           "Xor":
               {"n_downstream": 2,
                "downstream": ["Exist", "IsSame", "NotIsSame", "And", "Or", "Xor"],
                "sample_dist": [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
                "same_children_op": False
                },
           }
op_dict = defaultdict(dict, **op_dict)


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


def switch_generator(do_if, do_else, count):
    do_if_graph, do_if_root, do_if_node = do_if
    do_else_graph, do_else_root, do_else_node = do_else

    G: nx.DiGraph = nx.compose_all([do_if_graph, do_else_graph])
    switch_count = count
    G.add_node(switch_count, label='Switch')
    G.add_edge(do_if_node, switch_count)
    G.add_edge(do_else_node, switch_count)
    count += 1

    conditional_graph, conditional_root, count = subtask_graph_generator(max_depth=10, count=count)
    G = nx.compose(G, conditional_graph)
    G.add_edge(switch_count, conditional_root)
    return G, switch_count, count


def sample_children_helper(op_count, op_name, max_op):
    if op_count + 1 > max_op:
        return np.random.choice(op_dict[op_name]["downstream"], p=op_dict[op_name]["sample_dist"])
    else:
        return np.random.choice(op_dict[op_name]["downstream"])


def sample_children_op(op_count, op_name, max_depth, select_op, select_downstream):
    n_downstream = op_dict[op_name]["n_downstream"]

    if select_downstream is None:
        select_downstream = op_dict['Select']['downstream']

    if n_downstream == 1:
        return [random.choice(op_dict[op_name]["downstream"])]
    elif op_name == 'Select':
        if op_count + 1 > max_depth:
            return ['None' for _ in range(n_downstream)]
        else:
            ops = list()
            # if select at least one op
            if select_op:
                # remove a random Get op to be child of select
                ops.append(select_downstream.pop(random.randrange(len(select_downstream))))
                n_downstream -= 1
            for _ in range(n_downstream):
                if np.random.random() < 0.8:
                    ops.append('None')
                else:
                    if select_downstream:
                        ops.append(select_downstream.pop(random.randrange(len(select_downstream))))
                    else:
                        ops.append('None')
            return ops
    else:
        if op_dict[op_name]["same_children_op"]:
            child = sample_children_helper(op_count, op_name, max_depth)
            return [child for _ in range(n_downstream)]
        else:
            return [sample_children_helper(op_count, op_name, max_depth) for _ in range(n_downstream)]


def branch_generator(G, node, count, depth, max_depth, select_op=False, select_downstream=None):
    if node == 'None':
        return None
    elif node == 'CONST':
        return count
    else:
        if node == 'Exist':
            select_op = True

        children = sample_children_op(count, node, max_depth, select_op, select_downstream)
        if node == 'Select' and any('Get' in c for c in children):
            select_downstream = ['None'] * 4
        depth += 1
        parent = count
        for op in children:
            if op != 'None':
                child = count + 1
                count = branch_generator(G, op, count + 1, depth, max_depth, select_op, select_downstream)

                G.add_node(child, label=op)
                G.add_edge(parent, child)
        return count


def subtask_graph_generator(max_depth=10, count=1, select_limit=True):
    G = nx.DiGraph()
    root = count

    op_count = count
    root_op = random.choice(root_ops)
    G.add_node(op_count, label=root_op)
    op_queue = [(root, root_op)]

    while op_queue:
        (cur_count, cur_op) = op_queue.pop()
        children = sample_children_op(cur_count, cur_op, max_depth, select_limit)
        for op in children:
            if cur_op == 'Select':
                if not op == 'None':
                    op_count += 1
                    G.add_node(op_count, label=op)
                    attr = op.split('Get')[1]
                    G.add_edge(cur_count, op_count, label=attr)
                    op_queue.append((op_count, op))
            else:
                op_count += 1
                G.add_node(op_count, label=op)
                G.add_edge(cur_count, op_count)
                if op not in ['None', 'CONST']:
                    op_queue.append((op_count, op))

    return G, root, op_count


if __name__ == '__main__':
    # easy task repo
    stask_repo = []
    task_repo = []
    for _ in tqdm(range(100)):
        node_list, conn_mtx = subTask_Generator(max_op=6)
        # delete the task if the task is too big
        if len(node_list) > 20:
            continue
        stask_repo.append([node_list, conn_mtx])
    # sample from stask with switch op
    thres = 0.5
    for _ in tqdm(range(100)):
        if np.random.uniform() < thres:
            [task1, task2, task3] = random.choices(stask_repo, k=3)
            new_task = switch_generator(task1, task2, task3)
            task_repo.append(new_task)
            stask_repo.append(new_task)
        else:
            task_repo.append(random.choice(stask_repo))

    # save and draw tasks in task_repo
    print("drawing and saving------------")
    for i, task in enumerate(tqdm(task_repo)):
        [node_list, conn_mtx] = task
        G = nx.from_numpy_matrix(conn_mtx, create_using=nx.DiGraph)
        node_dict = {}
        for j in range(len(node_list)):
            node_dict[j] = {"label": node_list[j]}
        nx.set_node_attributes(G, node_dict)
        A = nx.nx_agraph.to_agraph(G)
        # TODO: inverse directions
        A.draw("autoTask/attributes_%d.png" % i, prog="dot")

    with open("autoTask/task_info.pkl", "wb") as fp:  # Pickling
        pickle.dump(task_repo, fp)
