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

"""A bank of available tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import random
import tensorflow.compat.v1 as tf

from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive.task_generator import Task
from cognitive.task_generator import TemporalTask
from cognitive import constants as const

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('task_family', 'all', 'name of the task to be trained')

GLOBAL_SEED = None


def check_whens(whens):
    while len(set(whens)) != len(whens):
        whens = sg.sample_when(len(whens))
    return whens


class GoShapeTemporal(TemporalTask):
    """Go to shape X."""

    def __init__(self, select_op_set=None):
        super(GoShapeTemporal, self).__init__()
        self.select_collection = []
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        inherent_attr = {"shape": shape1,
                         "when": when1}

        self._operator = tg.Go(objs1)
        select_tuple = [when1, inherent_attr, objs1, self._operator, "Go"]
        self.select_collection.append(select_tuple)

        self.n_frames = const.compare_when([when1]) + 1

        # update epoch index
        for select_op in self.select_collection:
            select_op[0] = self.n_frames - 1 - const.ALLWHENS.index(select_op[0])

    # def reinit(self, select_epoch_index, restrictions):
    #     for i, epoch_index in enumerate(select_epoch_index):
    #         select_index = self.op_index(epoch_index, )
    #         # update inherent attributes based on restrictions
    #         self.select_collection[select_index][2].update(self.select_collection[select_index][1], restrictions[i])
    #         # update _operator
    #         if self.select_collection[select_index][4] == "Go":
    #             self.select_collection[select_index][3] = tg.Go(self.select_collection[select_index][2])
    #
    # def op_index(self, epoch_index):
    #     for i, select_op in enumerate(self.select_collection):
    #         if epoch_index == select_op[0]:
    #             return i
    #     return "no select operator is found"

    @property
    def instance_size(self):
        return sg.n_random_shape() * sg.n_random_when()


class GoShape(TemporalTask):
    """Go to shape X."""

    def __init__(self):
        super(GoShape, self).__init__()
        shape1 = sg.random_shape()
        when1 = sg.random_when()
        objs1 = tg.Select(shape=shape1, when=when1)
        self._operator = tg.Go(objs1)

        self.n_frames = const.compare_when([when1]) + 1

    @property
    def instance_size(self):
        return sg.n_random_shape() * sg.n_random_when()


class GoColorOfTemporal(TemporalTask):
    """Go to shape 1 with the same color as shape 2.

    In general, this task can be extremely difficult, requiring memory of
    locations of all latest shape 2 of different colors.

    To make this task reasonable, we use customized generate_objset.

    Returns:
      task: task
    """

    def __init__(self):
        super(GoColorOfTemporal, self).__init__()
        shape1, shape2, shape3 = sg.sample_shape(3)
        # when1 = 'last%d' % (const.LASTMAP[sg.random_when(GLOBAL_SEED)] + 4)
        when1 = sg.random_when(GLOBAL_SEED)
        objs1 = tg.Select(shape=shape1, when=when1)
        color1 = tg.GetColor(objs1)
        objs2 = tg.Select(color=color1, shape=shape2, when='last0')
        self._operator = tg.Go(objs2)
        self._shape1, self._shape2, self._shape3, self._color1 = shape1, shape2, shape3, color1
        self.n_frames = const.compare_when([when1]) + 1

    def generate_objset(self, n_distractor=0, average_memory_span=3):
        objset = super(GoColorOfTemporal, self).generate_objset(n_distractor, average_memory_span)
        obj = next(iter(objset))
        shape = random.choice([self._shape1, self._shape2, self._shape3])
        test2 = sg.Object([shape, sg.another_color(obj.color)], when='last0')
        objset.add(test2, epoch_now=self.n_frames - 1)
        return objset

    @property
    def instance_size(self):
        return sg.n_sample_shape(3)


class GoShapeOfTemporal(TemporalTask):
    """Go to color 1 with the same shape as color 2."""

    def __init__(self):
        super(GoShapeOfTemporal, self).__init__()
        color1, color2, color3 = sg.sample_color(3)
        # when1 = 'last%d' % (const.LASTMAP[sg.random_when(GLOBAL_SEED)] + 4)
        when1 = sg.random_when(GLOBAL_SEED)
        objs1 = tg.Select(color=color1, when=when1)
        shape1 = tg.GetShape(objs1)
        objs2 = tg.Select(color=color2, shape=shape1, when='last0')
        self._operator = tg.Go(objs2)
        self._color1, self._color2, self._color3 = color1, color2, color3
        self.n_frames = const.compare_when([when1]) + 1

    def generate_objset(self, n_distractor=0, average_memory_span=3):
        objset = super(GoShapeOfTemporal, self).generate_objset(n_distractor, average_memory_span)
        obj = next(iter(objset))
        color = random.choice([self._color1, self._color2, self._color3])
        test2 = sg.Object([color, sg.another_shape([obj.shape])], when='last0')
        objset.add(test2, epoch_now=self.n_frames - 1)
        return objset

    @property
    def instance_size(self):
        return sg.n_sample_color(3)


class ExistCategoryOfTemporal(TemporalTask):
    """Check if exist object with shape of a colored object."""

    def __init__(self):
        super(ExistCategoryOfTemporal, self).__init__()
        when1 = sg.random_when()
        when2 = 'last0'

        objs1 = tg.Select(when=when1)
        category1 = tg.GetCategory(objs1)
        objs2 = tg.Select(category=category1, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class ExistViewAngleOfTemporal(TemporalTask):
    """Check if exist object with shape of a colored object."""

    def __init__(self):
        super(ExistViewAngleOfTemporal, self).__init__()
        when1 = sg.random_when()
        when2 = 'last0'

        objs1 = tg.Select(when=when1)
        view_angle = tg.GetViewAngle(objs1)
        objs2 = tg.Select(view_angle=view_angle, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class ExistObjectOfTemporal(TemporalTask):
    """Check if exist object with shape of a colored object."""

    def __init__(self):
        super(ExistViewAngleOfTemporal, self).__init__()
        when1 = sg.random_when()
        when2 = 'last0'
        objs1 = tg.Select(when=when1)
        obj = tg.GetObject(objs1)
        objs2 = tg.Select(object=obj, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class CompareCategoryTemporal(TemporalTask):
    """Compare category between two objects."""

    def __init__(self):
        super(CompareCategoryTemporal, self).__init__()
        when1, when2 = check_whens(sg.sample_when(2))
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetCategory(objs1)
        a2 = tg.GetCategory(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, 'last0']) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class CompareViewAngleTemporal(TemporalTask):
    """Compare view_angle between two objects."""

    def __init__(self):
        super(CompareViewAngleTemporal, self).__init__()
        when1, when2 = check_whens(sg.sample_when(2))
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetViewAngle(objs1)
        a2 = tg.GetViewAngle(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * (sg.n_random_when()) ** 2


class CompareLocTemporal(TemporalTask):
    """Compare color between two objects."""

    # TODO: compare loc has error: could not broadcast input array from shape (56,56,3) into shape (56,46,3)
    def __init__(self):
        super(CompareLocTemporal, self).__init__()
        when1, when2 = check_whens(sg.sample_when(2))
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetLoc(objs1)
        a2 = tg.GetLoc(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class CompareObjectTemporal(TemporalTask):
    """Compare color between two objects."""

    def __init__(self):
        super(CompareObjectTemporal, self).__init__()
        when1, when2 = check_whens(sg.sample_when(2))
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetObject(objs1)
        a2 = tg.GetObject(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


# class GoShapeTemporalComposite(tg.TemporalCompositeTask):
#     def __init__(self, n_tasks):
#         tasks = [GoShapeTemporal() for i in range(n_tasks)]
#         super(GoShapeTemporalComposite, self).__init__(tasks)
#         self.n_frames = sum([task.n_frames for task in tasks])


task_family_dict = OrderedDict([
    # ('CompareLoc', CompareLocTemporal),
    # ('ExistCategoryOf', ExistCategoryOfTemporal),
    # ('ExistViewAngleOf', ExistViewAngleOfTemporal),
    # ('ExistObjectOf', ExistObjectOfTemporal),
    # ('CompareViewAngle', CompareViewAngleTemporal),
    # ('CompareCategory', CompareCategoryTemporal),
    ('CompareObject', CompareObjectTemporal),
])


def random_task(task_family):
    """Return a random question from the task family."""
    return task_family_dict[task_family[random.randint(0, len(task_family) - 1)]]()
