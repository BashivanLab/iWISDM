# code based on https://github.com/google/cog

"""A bank of available tasks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import random

from iwisdm.envs.shapenet.task_generator import TemporalTask
from iwisdm.envs.shapenet import task_generator as tg
import iwisdm.envs.shapenet.stim_generator as sg
import iwisdm.envs.shapenet.registration as env_reg


class ExistCategoryOfTemporal(TemporalTask):
    """Check if  object on given frame has the same category with the object on another frame."""

    def __init__(self, whens, first_shareable=None):
        super(ExistCategoryOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        category1 = tg.GetCategory(objs1)
        objs2 = tg.Select(category=category1, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class ExistViewAngleOfTemporal(TemporalTask):
    """Check if  object on given frame has the same view angle with the object on another frame."""

    def __init__(self, whens, first_shareable=None):
        super(ExistViewAngleOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = whens[0], whens[1]

        objs1 = tg.Select(when=when1)
        view_angle = tg.GetViewAngle(objs1)
        objs2 = tg.Select(view_angle=view_angle, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class ExistObjectOfTemporal(TemporalTask):
    """Check if on given frame has the same object with another frame"""

    def __init__(self, whens, first_shareable=None):
        super(ExistObjectOfTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        obj = tg.GetObject(objs1)
        objs2 = tg.Select(object=obj, when=when2)
        self._operator = tg.Exist(objs2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class CompareCategoryTemporal(TemporalTask):
    """Compare objects on chosen frames are of the same category or not."""

    def __init__(self, whens, first_shareable=None):
        super(CompareCategoryTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetCategory(objs1)
        a2 = tg.GetCategory(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class CompareViewAngleTemporal(TemporalTask):
    """Compare objects on chosen frames are of the same view angle or not"""

    def __init__(self, whens, first_shareable=None):
        super(CompareViewAngleTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetViewAngle(objs1)
        a2 = tg.GetViewAngle(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class CompareLocTemporal(TemporalTask):
    """Compare objects on chosen frames are of the same location or not"""

    def __init__(self, whens, first_shareable=None):
        super(CompareLocTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetLoc(objs1)
        a2 = tg.GetLoc(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class CompareObjectTemporal(TemporalTask):
    """Compare objects on chosen frames are of the same identity or not"""

    def __init__(self, whens, first_shareable=None):
        super(CompareObjectTemporal, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)
        a1 = tg.GetObject(objs1)
        a2 = tg.GetObject(objs2)
        self._operator = tg.IsSame(a1, a2)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


class SequentialCategoryMatch(TemporalTask):
    # nback category
    def __init__(self, whens, first_shareable=None, n_compare=1):
        super(SequentialCategoryMatch, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]
        max_k = env_reg.compare_when([when1, when2])
        if n_compare * 2 + 1 > max_k:
            n_compare = max_k // 2
        total_frames = n_compare * 2 + random.randint(0, max_k - (n_compare * 2) + 1)

        sample_objs = [tg.Select(when=f'last{total_frames - i - 1}') for i in range(n_compare)]
        response_objs = [tg.Select(when=f'last{i}') for i in range(n_compare)]
        sample_cats = [tg.GetCategory(obj) for obj in sample_objs]
        response_cats = [tg.GetCategory(obj) for obj in response_objs]
        is_sames = [tg.IsSame(sample, response) for sample, response in zip(sample_cats, response_cats)]
        if n_compare == 1:
            self._operator = is_sames[0]
        else:
            ands = tg.And(is_sames[0], is_sames[1])
            for is_same1, is_same2 in zip(is_sames[2::2], is_sames[3::2]):
                ands = tg.And(tg.And(is_same1, is_same2), ands)
            self._operator = ands
        self.n_frames = total_frames


class DelayedCDM(TemporalTask):
    # contextual decision making

    def __init__(self, whens=None, first_shareable=None, attrs=None):
        super(DelayedCDM, self).__init__(whens=whens, first_shareable=first_shareable)
        when1, when2 = self.whens[0], self.whens[1]

        objs1 = tg.Select(when=when1)
        objs2 = tg.Select(when=when2)

        if attrs:
            assert len(attrs) == 3
            attr0 = attrs[0]
            attrs = attrs[1::]
        else:
            attr0 = random.choice(env_reg.DATA.ATTRS)
            attrs = random.sample(env_reg.DATA.ATTRS, 2)
        const_attr0 = sg.random_attr(attr0)
        const_attrs = [sg.random_attr(attr) for attr in attrs]
        condition = tg.IsSame(const_attr0, tg.get_family_dict[attr0](objs1))
        do_if = tg.IsSame(const_attrs[0], tg.get_family_dict[attrs[0]](objs2))
        do_else = tg.IsSame(const_attrs[1], tg.get_family_dict[attrs[1]](objs2))
        self._operator = tg.Switch(condition, do_if, do_else, both_options_avail=False)
        self.n_frames = env_reg.compare_when([when1, when2]) + 1


task_family_dict = OrderedDict([
    ('ExistCategoryOf', ExistCategoryOfTemporal),
    ('ExistViewAngleOf', ExistViewAngleOfTemporal),
    ('ExistObjectOf', ExistObjectOfTemporal),
    ('CompareViewAngle', CompareViewAngleTemporal),
    ('CompareCategory', CompareCategoryTemporal),
    ('CompareObject', CompareObjectTemporal),
    ('CompareLoc', CompareLocTemporal),
    ('SequentialCategory', SequentialCategoryMatch),
    ('DelayedCDM', DelayedCDM),
])
