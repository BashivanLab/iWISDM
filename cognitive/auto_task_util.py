import numpy as np
import random
import networkx as nx
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# TODO: add select attributes, combine helper with task_generator.task_generation
root_op = ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", "Exist", "Equal", "And", "Or", "Xor"]
leaf_op = ["Select"]
mid_op = ["Switch"]

# TODO: And can have different children, And:{XOR, Equal}
op_dict = {"Select":
               {"n_downstream": 0,
                "downstream": [], },
           "GetCategory":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]},
           "GetLoc":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]},
           "GetViewAngle":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]},
           "GetObject":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]},
           "Exist":
               {"n_downstream": 1,
                "downstream": ["Select"],
                "sample_dist": [1]},
           "Equal":
               {"n_downstream": 2,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", ],
                "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                "IsSame": True, },  # same downstream op
           "NotEqual":
               {"n_downstream": 2,
                "downstream": ["GetCategory", "GetLoc", "GetViewAngle", "GetObject", ],
                "sample_dist": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                "IsSame": True, },  # same downstream op
           "And":
               {"n_downstream": 2,
                "downstream": ["Exist", "Equal", "NotEqual", "And", "Or", "Xor"],
                "sample_dist": [2 / 9, 2 / 9, 2 / 9, 1 / 9, 1 / 9, 1 / 9],
                "IsSame": False},
           "Or":
               {"n_downstream": 2,
                "downstream": ["Exist", "Equal", "NotEqual", "And", "Or", "Xor"],
                "sample_dist": [2 / 9, 2 / 9, 2 / 9, 1 / 9, 1 / 9, 1 / 9],
                "IsSame": False},
           "Xor":
               {"n_downstream": 2,
                "downstream": ["Exist", "Equal", "NotEqual", "And", "Or", "Xor"],
                "sample_dist": [2 / 9, 2 / 9, 2 / 9, 1 / 9, 1 / 9, 1 / 9],
                "IsSame": False},
           }


def SwitchTask_Generator(task1, task2, task3):
    nl1, cm1 = task1
    nl2, cm2 = task2
    nl3, cm3 = task3
    l1 = len(nl1)
    l2 = len(nl2)
    l3 = len(nl3)
    node_list = nl1 + nl2 + ["switch"] + nl3
    conn_mtx = np.zeros((l1 + l2 + l3 + 1, l1 + l2 + l3 + 1))
    conn_mtx[:l1, :l1] = cm1
    conn_mtx[l1:l1 + l2, l1:l1 + l2] = cm2
    conn_mtx[l1 - 1, l1 + l2] = 1
    conn_mtx[l1 + l2 - 1, l1 + l2] = 1
    conn_mtx[l1 + l2 + 1:, l1 + l2 + 1:] = cm3
    conn_mtx[l1 + l2, l1 + l2 + 1] = 1
    return [node_list, conn_mtx]


def subTask_Generator(max_op=32, ):
    conn_mtx = np.zeros((3 ** max_op, 3 ** max_op))
    node_list = []
    root_node = random.choice(root_op)
    node_list.append(root_node)
    curr_node = root_node
    curr_node_idx = 0
    pos_node_idx = 0
    done = False

    while not done:  ## how to constraint the number of operators?
        curr_node = node_list[curr_node_idx]
        if op_dict[curr_node]["n_downstream"] == 1:
            if pos_node_idx > max_op: # select operators based on sample_dist to limit the depth of the tree
                curr_node = random.choices(op_dict[curr_node]["downstream"], op_dict[curr_node]["sample_dist"])[0]
            else:
                curr_node = random.choice(op_dict[curr_node]["downstream"])
            node_list.append(curr_node)
            conn_mtx[curr_node_idx, pos_node_idx + 1] = 1
            curr_node_idx += 1
            pos_node_idx += 1
        elif op_dict[curr_node]["n_downstream"] == 2:
            if pos_node_idx > max_op:
                curr_node = random.choices(op_dict[curr_node]["downstream"], op_dict[curr_node]["sample_dist"])[0]
            else:
                curr_node = random.choice(op_dict[curr_node]["downstream"])
            node_list.append(curr_node)
            node_list.append(curr_node)
            conn_mtx[curr_node_idx, pos_node_idx + 1] = 1
            conn_mtx[curr_node_idx, pos_node_idx + 2] = 1
            curr_node_idx += 1
            pos_node_idx += 2
        elif op_dict[curr_node]["n_downstream"] == 0:
            if all([op == "Select" for op in node_list[curr_node_idx:pos_node_idx + 1]]):
                done = True
            else:
                curr_node_idx += 1

    conn_mtx = conn_mtx[:pos_node_idx + 1, :pos_node_idx + 1]
    return node_list, conn_mtx


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
            new_task = SwitchTask_Generator(task1, task2, task3)
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
