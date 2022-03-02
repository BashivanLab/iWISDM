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

from cognitive import constants as const
from cognitive import stim_generator as sg
from cognitive import task_generator as tg


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


if __name__ == '__main__':
    unittest.main()
