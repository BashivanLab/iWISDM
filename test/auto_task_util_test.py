from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx
import numpy as np
import random

from cognitive import constants as const
from cognitive.auto_task import auto_task_util as util
import cognitive.task_generator as tg


class UtilTest(unittest.TestCase):
    def test_sample_children_op(self):
        samples = [util.sample_children_op(0, 'Select', 10) for _ in range(10)]

    def test_subtask_graph_generator(self):
        G, root, op_count = util.subtask_graph_generator(10, select_limit=True)
        A = nx.nx_agraph.to_agraph(G)
        A.draw("subtask.png", prog="dot")

    def test_Relabel_Graph(self):
        do_if_graph, _, _ = util.subtask_graph_generator()
        print(do_if_graph.nodes(data='label'))
        relabeled_graph = util.relabel_graph(do_if_graph, 20)
        print(relabeled_graph.nodes(data='label'))

    def test_switch_generator(self):
        do_if = util.subtask_graph_generator()
        do_else = util.subtask_graph_generator(count=do_if[2] + 1)
        conditional = util.subtask_graph_generator(count=do_else[2] + 1)

        G, switch_count, count = util.switch_generator(conditional, do_if, do_else)
        # A = nx.nx_agraph.to_agraph(do_if[0])
        # A.draw("do_if.png", prog="dot")
        # A = nx.nx_agraph.to_agraph(do_else[0])
        # A.draw("do_else.png", prog="dot")

        A = nx.nx_agraph.to_agraph(G)
        A.draw("switch.png", prog="dot")
        G = G.reverse()
        A = nx.nx_agraph.to_agraph(G)
        A.draw("switch_reverse.png", prog="dot")

    def test_branch_generator(self):
        G = nx.DiGraph()
        G.add_node(1, label='And')
        util.branch_generator(G, 'And', 1, 20, 0, 10)
        print(G.nodes(data=True))
        A = nx.nx_agraph.to_agraph(G)
        A.draw("branch.png", prog="dot")

    def test_write_task(self):
        const.DATA = const.Data()
        G = nx.DiGraph()
        G.add_node(1, label='IsSame')
        G.add_node(2, label='GetViewAngle')
        G.add_node(3, label='GetViewAngle')
        G.add_node(4, label='Select')
        G.add_node(5, label='Select')

        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 5)

        op, task = tg.subtask_generation((G, 1, 5))
        print(task)
        G = G.reverse()
        A = nx.nx_agraph.to_agraph(G)
        A.draw('CompareViewAngle.png', prog='dot')
        task.to_json('view_angle.json')

    def test_write_switch_task(self):
        G = nx.DiGraph()
        G.add_node(1, label='IsSame')
        G.add_node(2, label='GetCategory')
        G.add_node(3, label='GetCategory')
        G.add_node(4, label='Select')
        G.add_node(5, label='Select')
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 4)
        G.add_edge(3, 5)

        G.add_node(6, label='IsSame')
        G.add_node(7, label='GetObject')
        G.add_node(8, label='GetObject')
        G.add_node(9, label='Select')
        G.add_node(10, label='Select')
        G.add_edge(6, 7)
        G.add_edge(6, 8)
        G.add_edge(7, 9)
        G.add_edge(8, 10)

        G.add_node(11, label='IsSame')
        G.add_node(12, label='GetLoc')
        G.add_node(13, label='GetLoc')
        G.add_node(14, label='Select')
        G.add_node(15, label='Select')
        G.add_edge(11, 12)
        G.add_edge(11, 13)
        G.add_edge(12, 14)
        G.add_edge(13, 15)

        G.add_node(16, label='Switch')
        G.add_edge(5, 16)
        G.add_edge(9, 16)
        G.add_edge(16, 11)
        G = G.reverse()
        A = nx.nx_agraph.to_agraph(G)
        A.draw('RandomTask1.png', prog='dot')

    def test_task_generation(self):
        const.DATA = const.Data(
            dir_path='/Users/markbai/Documents/COG_v3_shapenet/data/new_shapenet_val'
        )
        graph, task = util.task_generator(0, 1, 10, 3, True)
        G = graph[0]
        G = G.reverse()
        A = nx.nx_agraph.to_agraph(G)
        A.draw('RandomTask1.png', prog='dot')
        G = task[1].to_graph()


if __name__ == '__main__':
    unittest.main()
