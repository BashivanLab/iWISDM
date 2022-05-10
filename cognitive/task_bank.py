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


class SimpleExistObjGo(Task):
    """If exist color A then go color A, else go color B."""

    # TODO: try both_options_avail for temp switch, where t2 and t3 both appear
    def __init__(self, whens=None, first_shareable=None):
        super(SimpleExistObjGo, self).__init__(whens=whens, first_shareable=first_shareable)
        obj1, obj2, obj3 = sg.sample_object(2, category=sg.random_category())
        when1, when2, when3 = sg.check_whens(sg.sample_when(3))
        objs1 = tg.Select(obj=obj1, when=when1)
        objs2 = tg.Select(obj=obj2, when=when2)
        objs3 = tg.Select(obj=obj2, when=when3)
        self._operator = tg.Switch(
            tg.Exist(objs1), tg.Go(objs2), tg.Go(objs3), both_options_avail=True)
        self.n_frames = const.compare_when([when1, when2, when3])

    @property
    def instance_size(self):
        return sg.n_sample_color(2) * sg.n_random_when()


class SimpleExistCatGo(TemporalTask):
    """If exist cat A then go cat A, else go cat B."""

    def __init__(self, whens=None, first_shareable=None):
        super(SimpleExistCatGo, self).__init__(whens=whens, first_shareable=first_shareable)
        cat1, cat2, cat3 = sg.sample_category(3)
        if self.whens is not None:
            when0, when1, when2 = sg.check_whens(sg.sample_when(3))
        else:
            when0, when1, when2 = self.whens[0], self.whens[1], self.whens[2]
        objs1 = tg.Select(category=cat1, when=when0)
        objs2 = tg.Select(when=when1)
        objs3 = tg.Select(when=when2)
        self._operator = tg.Switch(
            tg.Exist(objs1), tg.Go(objs2), tg.Go(objs3), both_options_avail=False)
        self.n_frames = const.compare_when([when0, when1, when2]) + 1

    @property
    def instance_size(self):
        return sg.n_sample_shape(2) * sg.n_random_when()


task_family_dict = OrderedDict([
    ('ExistCategoryOf', ExistCategoryOfTemporal),
    ('ExistViewAngleOf', ExistViewAngleOfTemporal),
    ('ExistObjectOf', ExistObjectOfTemporal),
    ('CompareViewAngle', CompareViewAngleTemporal),
    ('CompareCategory', CompareCategoryTemporal),
    ('CompareFixedObject', CompareFixedObjectTemporal),
    ('CompareObject', CompareObjectTemporal),
    ('CompareLoc', CompareLocTemporal),
    ('SimpleExistCatGo', SimpleExistCatGo)
])


def random_task(task_family, when1, when2):
    """Return a random question from the task family."""
    return task_family_dict[task_family[random.randint(0, len(task_family) - 1)]](when1, when2)
