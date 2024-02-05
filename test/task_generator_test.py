# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for cognitive/task_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import unittest
import pickle
import networkx as nx
import sys

from cognitive import constants as const
from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive.auto_task import auto_task_util as util


def targets_to_str(targets):
    return [t.value if hasattr(t, 'value') else str(t) for t in targets]


class TaskGeneratorTest(unittest.TestCase):
    def testGetLeafs(self):
        G, root, op_count = util.subtask_graph_generator()
        leafs = tg.get_leafs(G)
        print([G.nodes[node]['label'] for node in leafs])

    def testConvertOperators(self):
        const.DATA = const.Data()
        G, root, op_count = util.subtask_graph_generator()
        A = nx.nx_agraph.to_agraph(G)
        A.draw("convert.png", prog="dot")
        operators = {node[0]: node[1]['label'] for node in G.nodes(data=True)}

        operator_families = tg.get_operator_dict()
        selects = [op for op in G.nodes() if operators[op] == 'Select']
        const.DATA.MAX_MEMORY = len(selects) + 1
        whens = sg.check_whens(sg.sample_when(len(selects)))
        n_frames = const.compare_when(whens) + 1

        whens = {select: when for select, when in zip(selects, whens)}
        op = tg.graph_to_operators(G, root, operators, operator_families, whens)
        task = tg.TemporalTask(operator=op, n_frames=n_frames)
        print(task)

    def testSubTaskGeneration(self):
        const.DATA = const.Data()
        subtask = util.subtask_graph_generator()
        print(tg.subtask_generation(subtask)[1])

    def testSwitchGeneration(self):
        const.DATA = const.Data()
        do_if = util.subtask_graph_generator()
        do_if_task = tg.subtask_generation(do_if)

        do_else = util.subtask_graph_generator(count=do_if[2] + 1)
        do_else_task = tg.subtask_generation(do_else)

        conditional = util.subtask_graph_generator(count=do_else[2] + 1)
        conditional_task = tg.subtask_generation(conditional)

        G, _, _ = util.switch_generator(conditional, do_if, do_else)
        A = nx.nx_agraph.to_agraph(G)
        A.draw("convert_switch.png", prog="dot")
        op, task = tg.switch_generation(conditional_task, do_if_task, do_else_task)
        with open('data.pkl', 'wb') as h:
            pickle.dump(conditional_task[0], h)

    def test_load_json(self):
        const.DATA = const.Data()
        with open('test.json', 'r') as f:
            op = json.load(f)

        print(tg.load_operator_json(op, tg.get_operator_dict(), tg.get_attr_dict()))

    def test_to_json(self):
        const.DATA = const.Data()
        c = sg.sample_category(1)
        o = sg.sample_object(1, c[0])
        v = sg.sample_view_angle(1, o[0])
        get1 = tg.GetCategory(tg.Select(category=c))
        get2 = sg.sample_category(1)[0]
        statement = tg.IsSame(get1, get2)
        do_if_true = tg.GetViewAngle(tg.Select(view_angle=v))
        do_if_false = tg.GetObject(tg.Select(object=o))
        op = tg.Switch(statement, do_if_true, do_if_false)
        task = tg.Task(operator=op)

        objset = task.generate_objset(n_epoch=5)
        task.get_target(objset)
        print(op.__class__.__name__, op.child[0].__class__.__name__)
        print(op.to_json())
        with open('test.json', 'w') as f:
            json.dump(op.to_json(), f, indent=4)

    def testAddObj(self):
        const.DATA = const.Data()
        obj = sg.Object(when='last0')
        objset = sg.ObjectSet(n_epoch=5)
        temp = objset.add(obj, epoch_now=0).copy()
        for i in range(1, 5):
            temp = objset.add(temp, epoch_now=i).copy()
        print(objset)

    def testAttrStr(self):
        const.DATA = const.Data(
            dir_path='/Users/markbai/Documents/COG_v3_shapenet/data/new_shapenet_val/',
            train=False
        )
        categories = sg.sample_category(4)
        objects = [sg.sample_object(k=1, category=cat)[0] for cat in categories]
        view_angles = [sg.sample_view_angle(k=1, obj=obj)[0] for obj in objects]

        op1 = tg.Select(when='last2')
        op2 = tg.Select(category=categories[1], when='last1')
        new_task0 = tg.TemporalTask(tg.GetCategory(op1), 1)
        objset = new_task0.generate_objset()

        print(new_task0.get_target(objset))
        new_task1 = tg.TemporalTask(tg.IsSame(tg.GetCategory(op1), tg.GetCategory(op2)), 3)


if __name__ == '__main__':
    unittest.main()
