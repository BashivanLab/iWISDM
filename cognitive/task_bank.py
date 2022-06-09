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
from cognitive.task_generator import TemporalTask
from cognitive import constants as const

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('task_family', 'all', 'name of the task to be trained')

GLOBAL_SEED = None


class ExistCategoryOfTemporal(TemporalTask):
    """Check if exist object with shape of a colored object."""

    def __init__(self, whens=None, first_shareable=None):
        super(ExistCategoryOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
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

    def __init__(self, whens=None, first_shareable=None):
        super(ExistViewAngleOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
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

    def __init__(self, whens=None, first_shareable=None):
        super(ExistViewAngleOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        obj = tg.GetObject(objs1)
        objs2 = tg.Select(obj=obj, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class CompareCategoryTemporal(TemporalTask):
    """DMS, Compare category between two objects."""

    def __init__(self, whens=None, first_shareable=None):
        super(CompareCategoryTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetCategory(objs1)
        a2 = tg.GetCategory(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class CompareViewAngleTemporal(TemporalTask):
    """Compare view_angle between two objects."""

    def __init__(self, whens=None, first_shareable=None):
        super(CompareViewAngleTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
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

    def __init__(self, whens=None, first_shareable=None):
        super(CompareLocTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
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

    def __init__(self, whens=None, first_shareable=None):
        super(CompareObjectTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetObject(objs1)
        a2 = tg.GetObject(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class CompareFixedObjectTemporal(TemporalTask):
    """Compare between two objects."""

    def __init__(self, whens=None, first_shareable=None):
        super(CompareFixedObjectTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = sg.check_whens(sg.sample_when(2))
        else:
            when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetFixedObject(objs1)
        a2 = tg.GetFixedObject(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = const.compare_when([when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class SequentialCategoryMatch(TemporalTask):
    def __init__(self, whens=None, first_shareable=None, n_frames=1):
        super(SequentialCategoryMatch, self).__init__(whens=whens, first_shareable=first_shareable)
        total_frames = n_frames * 2 + random.randint(0, const.MAX_MEMORY - (n_frames * 2) + 1)

        sample_objs = [tg.Select(when=f'last{total_frames - i - 1}') for i in range(n_frames)]
        response_objs = [tg.Select(when=f'last{i}') for i in range(n_frames)]
        sample_cats = [tg.GetCategory(obj) for obj in sample_objs]
        response_cats = [tg.GetCategory(obj) for obj in response_objs]
        is_sames = [tg.IsSame(sample, response) for sample, response in zip(sample_cats, response_cats)]
        if n_frames == 1:
            self._operator = is_sames[0]
        else:
            ands = tg.And(is_sames[0], is_sames[1])
            for is_same1, is_same2 in zip(is_sames[2::2], is_sames[3::2]):
                ands = tg.And(tg.And(is_same1, is_same2), ands)
            self._operator = ands
        self.n_frames = total_frames

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * (sg.n_random_when()) ** 2


class DelayedCDM(TemporalTask):
    def __init__(self, whens=None, first_shareable=None, attrs=None):
        super(DelayedCDM, self).__init__(whens=whens, first_shareable=first_shareable)
        if self.whens is None:
            when1, when2 = reversed(sorted(sg.check_whens(sg.sample_when(2))))
        else:
            when1, when2 = self.whens[0], self.whens[1]

        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        if attrs:
            assert len(attrs) == 3
            attr0 = attrs[0]
            attrs = attrs[1::]
        else:
            attr0 = random.choice(const.ATTRS)
            attrs = random.sample(const.ATTRS, 2)
        const_attr0 = sg.random_attr(attr0)
        const_attrs = [sg.random_attr(attr) for attr in attrs]
        condition = tg.IsSame(const_attr0, tg.get_family_dict[attr0](objs1))
        do_if = tg.IsSame(const_attrs[0], tg.get_family_dict[attrs[0]](objs2))
        do_else = tg.IsSame(const_attrs[1], tg.get_family_dict[attrs[1]](objs2))
        self._operator = tg.Switch(condition, do_if, do_else)
        self.n_frames = const.compare_when([when1, when2]) + 1

# add delayedCMS without constants

task_family_dict = OrderedDict([
    ('ExistCategoryOf', ExistCategoryOfTemporal),
    ('ExistViewAngleOf', ExistViewAngleOfTemporal),
    ('ExistObjectOf', ExistObjectOfTemporal),
    ('CompareViewAngle', CompareViewAngleTemporal),
    ('CompareCategory', CompareCategoryTemporal),
    ('CompareFixedObject', CompareFixedObjectTemporal),
    ('CompareObject', CompareObjectTemporal),
    ('CompareLoc', CompareLocTemporal),
    ('SequentialCategory', SequentialCategoryMatch),
    ('DelayedCDM', DelayedCDM)
])


def random_task(task_family, *args):
    """Return a random question from the task family."""
    return task_family_dict[task_family[random.randint(0, len(task_family) - 1)]](*args)
