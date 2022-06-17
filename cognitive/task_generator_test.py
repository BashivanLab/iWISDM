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

import unittest
import networkx as nx

from cognitive import constants as const
from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive.auto_task import auto_task_util as util


def targets_to_str(targets):
    return [t.value if hasattr(t, 'value') else str(t) for t in targets]


class TaskGeneratorTest(unittest.TestCase):
    def testSelectUpdate(self):
        shape1, shape2 = sg.sample_shape(2)
        while shape1 == shape2:
            shape1, shape2 = sg.sample_shape(2)
        color1, color2 = sg.sample_color(2)
        while color2 == color1:
            color1, color2 = sg.sample_color(2)
        when1 = sg.random_when()

        object1 = sg.Object([color2, shape2], when=when1)
        objs = tg.Select(shape=shape1, color=color1, when=when1)

        objs.update(object1)
        self.assertTrue(objs.shape == shape2)
        self.assertTrue(objs.color == color2)

    def testSelectReinit(self):
        for _ in range(1000):
            shapes = sg.sample_shape(3)
            while len(shapes) != len(set(shapes)):
                shapes = sg.sample_shape(3)
            colors = sg.sample_color(3)
            while len(colors) != len(set(colors)):
                colors = sg.sample_color(3)
            whens = sg.sample_when(2)
            while len(whens) != len(set(whens)):
                whens = sg.sample_when(3)
            object1 = sg.Object([colors[2], shapes[2]], when=whens[1])
            op = tg.Select(color=colors[1], shape=shapes[1], when=whens[1])
            task1 = tg.TemporalTask(tg.GetShape(op))
            task1.reinit([object1])

            self.assertTrue(op.shape == shapes[2])
            self.assertTrue(op.color == colors[2])

            object1 = sg.Object([colors[1], shapes[1]], when=whens[1])
            op1 = tg.Select(color=colors[2], shape=shapes[2], when=whens[1])
            op2 = tg.Select(color=colors[0], shape=shapes[0], when=whens[1])
            task1 = tg.TemporalTask(tg.IsSame(tg.GetShape(op1), tg.GetShape(op2)))
            self.assertFalse(task1.reinit([object1]))

            object1 = sg.Object([colors[1], shapes[1]], when=whens[1])
            op1 = tg.Select(color=colors[2], shape=shapes[2], when=whens[1])
            op2 = tg.Select(color=colors[0], shape=shapes[0], when=whens[0])
            task1 = tg.TemporalTask(tg.IsSame(tg.GetShape(op1), tg.GetShape(op2)))
            self.assertTrue(task1.reinit([object1]))

            object1 = sg.Object([colors[2], shapes[2]], when=whens[1])
            op = tg.Select(color=colors[1], shape=shapes[1], when=whens[1])
            task1 = tg.TemporalTask(tg.GetShape(op))
            task1.n_frames = 2
            task1.reinit([object1])
            objset = task1.generate_objset()
            target_values = [const.get_target_value(t) for t in task1.get_target(objset)]
            self.assertEqual(object1.shape.value, target_values[0])

    def testTaskGeneration(self):
        tg.task_generation()

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
        op = tg.convert_operators(G, root, operators, operator_families, whens)
        task = tg.TemporalTask(operator=op, n_frames=n_frames)
        print(task)

    def testSubTaskGeneration(self):
        const.DATA = const.Data()
        subtask = util.subtask_graph_generator()
        print(tg.subtask_generation(subtask)[1])

    def testSwitchGeneration(self):
        const.DATA = const.Data()
        do_if = util.subtask_graph_generator()
        do_else = util.subtask_graph_generator(count=do_if[2] + 1)
        conditional = util.subtask_graph_generator(count=do_else[2] + 1)
        G, _, _ = util.switch_generator(conditional, do_if, do_else)
        A = nx.nx_agraph.to_agraph(G)
        A.draw("convert_switch.png", prog="dot")
        print(tg.switch_generation(conditional, do_if, do_else))


if __name__ == '__main__':
    unittest.main()
