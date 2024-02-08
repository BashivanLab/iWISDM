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

"""High-level API for generating stimuli.

Objects are first generated abstractly, with high-level specifications
like location='random'.
Abstract relationships between objects can also be specified.

All objects and relationships are then collected into a ObjectSet.
The ObjectSet object can interpret the abstract specifications and instantiate
the stimuli in each trial.

Rendering function generates movies based on the instantiated stimuli
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from bisect import bisect_left
from collections import defaultdict
import random
from typing import List

import numpy as np
import cv2 as cv2

from cognitive import constants as const


def _get_space_to(x0: float, x1: float, y0: float, y1: float, space_type: str):
    """
    given the 2D coordinate and the , return the
    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param space_type:
    :return:
    """
    if space_type == 'right':
        space = [(x1, 0.95), (0.05, 0.95)]
    elif space_type == 'left':
        space = [(0.05, x0), (0.05, 0.95)]
    elif space_type == 'top':
        space = [(0.05, 0.95), (0.05, y0)]
    elif space_type == 'bottom':
        space = [(0.05, 0.95), (y1, 0.95)]
    else:
        raise ValueError('Unknown space type: ' + str(space_type))

    return Space(space)


class Attribute(object):
    """Base class for attributes."""

    def __init__(self, value):
        self.value = value if not isinstance(value, list) else tuple(value)
        self.parent = list()

    def __call__(self, *args):
        """Including a call function to be consistent with Operator class."""
        return self

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        """Override the default Equals behavior."""
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __ne__(self, other):
        """Define a non-equality test."""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return True

    def __hash__(self):
        return hash(self.value)

    def self_json(self):
        return {}

    def to_json(self):
        info = dict()
        info['name'] = self.__class__.__name__
        info['value'] = self.get_value
        info.update(self.self_json())
        return info

    def resample(self):
        raise NotImplementedError('Abstract method.')

    @property
    def has_value(self):
        return self.value is not None

    @property
    def get_value(self):
        return self.value


class Loc(Attribute):
    """Location class."""

    def __init__(self, space=None, value=None):
        """Initialize location.

        Args:
          value: None or a tuple of floats
          space: None or a tuple of tuple of floats
          If tuple of floats, then the actual
        """
        super(Loc, self).__init__(value)
        if space is None:
            space = random_grid_space()
        self.attr_type = 'location'
        self.space = space

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.space == other.space
        return False

    def __str__(self):
        if len(const.DATA.grid) == 4:
            key = const.DATA.get_grid_key(self.space)
            if key[0] == 0:
                if key[1] == 0:
                    quadrant = 'top left'
                else:
                    quadrant = 'bottom left'
            else:
                if key[1] == 0:
                    quadrant = 'top right'
                else:
                    quadrant = 'bottom right'
            return '' + quadrant
        return 'location: ' + self.value

    def get_space_to(self, space_type):
        if self.value is None:
            return Space(None)
        else:
            x, y = self.value
            return _get_space_to(x, x, y, y, space_type)

    def get_opposite_space_to(self, space_type):
        opposite_space = {'left': 'right',
                          'right': 'left',
                          'top': 'bottom',
                          'bottom': 'top',
                          }[space_type]
        return self.get_space_to(opposite_space)

    @property
    def get_value(self):
        return self.value if self.has_value else self.value


class Space(Attribute):
    """Space class."""

    def __init__(self, value=None):
        super(Space, self).__init__(value)
        if self.value is None:
            self._value = [(0, 1), (0, 1)]
        else:
            self._value = value

    def sample(self, avoid=None):
        """Sample a location.

        This function will attempt to find a location to place the object
        that doesn't overlap will other objects at locations avoid,
        but will place an object anyway if it didn't find a good place

        Args:
          avoid: a list of locations (tuples) to be avoided
        """
        mid_point = (1 / 2, 1 / 2)
        if avoid is None:
            avoid = [mid_point]
        else:
            avoid = avoid + [mid_point]
        # avoid the mid-point for fixation cue

        n_max_try = 100
        avoid_radius2 = 0.04  # avoid radius squared

        dx = 0.001  # used to be 0.125 xuan => it does not matter now
        xrange = (self._value[0][0] + dx, self._value[0][1] - dx)
        dy = 0.001  # used to be 0.125 xuan => it does not matter now
        yrange = (self._value[1][0] + dy, self._value[1][1] - dy)
        for i_try in range(n_max_try):
            # Round to 3 decimal places to save space in json dump
            loc_sample = (round(random.uniform(*xrange), 3),
                          round(random.uniform(*yrange), 3))

            not_overlapping = True
            for loc_avoid in avoid:
                not_overlapping *= ((loc_sample[0] - loc_avoid[0]) ** 2 +
                                    (loc_sample[1] - loc_avoid[1]) ** 2 > avoid_radius2)

            if not_overlapping:
                break
        return Loc(space=self, value=loc_sample)

    def include(self, loc):
        """Check if an unsampled location (a space) includes a location."""
        x, y = loc.value
        return ((self._value[0][0] < x < self._value[0][1]) and
                (self._value[1][0] < y < self._value[1][1]))

    def get_space_to(self, space_type: str):
        x0, x1 = self._value[0]
        y0, y1 = self._value[1]
        return _get_space_to(x0, x1, y0, y1, space_type)

    def get_opposite_space_to(self, space_type: str):
        opposite_space = {'left': 'right',
                          'right': 'left',
                          'top': 'bottom',
                          'bottom': 'top',
                          }[space_type]
        return self.get_space_to(opposite_space)

    @property
    def get_value(self):
        return self._value if self.has_value else self.value


class SNCategory(Attribute):
    def __init__(self, value):
        super(SNCategory, self).__init__(value)
        self.attr_type = 'category'

    def sample(self):
        self.value = random_category().value

    def resample(self):
        self.value = another_category(self).value

    def __str__(self):
        if self.attr_type in const.DATA.mods_with_mapping:
            return '' + const.DATA.mods_with_mapping[self.attr_type][self.value]
        return 'category: ' + str(self.value)

    @property
    def get_value(self):
        return int(self.value) if self.has_value else self.value


class SNObject(Attribute):
    def __init__(self, category, value):
        if value is not None:
            assert isinstance(category, SNCategory)
        super(SNObject, self).__init__(value)
        self.attr_type = 'object'
        self.category = category

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value and self.category == other.category
        return False

    def __str__(self):
        if self.attr_type in const.DATA.mods_with_mapping:
            return '' + const.DATA.mods_with_mapping[self.attr_type][self.value]
        return 'object: ' + str(self.value)

    def sample(self):
        self.value = random_object(self.category).value

    def resample(self):
        self.value = another_object(self).value

    def self_json(self):
        return {'category': self.category.to_json()}

    @property
    def get_value(self):
        return int(self.value) if self.has_value else self.value


class SNViewAngle(Attribute):
    def __init__(self, sn_object, value):
        if value is not None:
            assert isinstance(sn_object, SNObject)
        super(SNViewAngle, self).__init__(value)
        self.attr_type = 'view_angle'
        self.object = sn_object

    def __str__(self):
        if self.attr_type in const.DATA.mods_with_mapping:
            return '' + const.DATA.mods_with_mapping[self.attr_type][self.value]
        return 'view_angle: ' + str(self.value)

    def sample(self):
        self.value = random_view_angle(self.object).value

    def resample(self):
        self.value = another_view_angle(self).value

    def self_json(self):
        return {'sn_object': self.object.to_json()}

    @property
    def get_value(self):
        return int(self.value) if self.has_value else self.value


class SNFixedObject(Attribute):
    # used for sanity check, fixed object, only changes view angle
    def __init__(self, sn_object, value):
        if value is not None:
            assert isinstance(sn_object, SNObject)
        super(SNFixedObject, self).__init__(value)
        self.attr_type = 'fixed_object'
        self.object = sn_object

    def __str__(self):
        return 'object: ' + str(self.value)

    def sample(self):
        self.value = random_view_angle(self.object).value

    def resample(self):
        self.value = another_view_angle(self).value

    @property
    def get_value(self):
        return int(self.value) if self.has_value else self.value


def static_objects_from_dict(d):
    epochs = d['epochs']
    epochs = epochs if isinstance(epochs, list) else [epochs]
    return [StaticObject(loc=tuple(d['location']),
                         category=d['category'],
                         object=d['object'],
                         view_angle=d['view_angle'],
                         epoch=e)
            for e in epochs]


class StaticObject(object):
    """Object that can be loaded from dataset and rendered."""

    def __init__(self, loc, category, object, view_angle, epoch):
        self.loc = loc  # 2-tuple of floats
        self.category = category  # string
        self.object = object  # string
        self.view_angle = view_angle
        self.epoch = epoch  # int


class StaticObjectSet(object):
    """Provides a subset of functionality provided by ObjectSet.

    This functionality is just enough to create StaticObjectSets from
    json strings and use them to generate feeds for training.
    """

    def __init__(self, n_epoch, static_objects=None, targets=None):
        self.n_epoch = n_epoch

        # {epoch -> [objs]}
        self.dict = defaultdict(list)
        if static_objects:
            for o in static_objects:
                self.add(o)

        self.targets = targets

    def add(self, obj):
        self.dict[obj.epoch].append(obj)

    def select_now(self, epoch_now):
        subset = self.dict[epoch_now]
        # Order objects by location to have a deterministic ordering.
        # Ordering determines occlusion.
        subset.sort(key=lambda o: (o.location, o.category, o.object, o.view_angle))
        return subset


class Object(object):
    """An object on the screen.

    An object is a collection of attributes.

    Args:
      location: tuple (x, y)
      color: string ('red', 'green', 'blue', 'white')
      shape: string ('circle', 'square')
      when: string ('last', 'last1', 'last2',)
      deletable: boolean. Whether or not this object is deletable. True if
        distractors.

    Raises:
      TypeError if location, color, shape are neither None nor respective Attributes
    """

    def __init__(self,
                 attrs=None,
                 when=None,
                 deletable=False):

        self.space = random_grid_space()
        self.location = Loc(space=self.space, value=None)
        self.category = SNCategory(value=None)
        self.object = SNObject(self.category, value=None)
        self.view_angle = SNViewAngle(self.object, value=None)

        if attrs is not None:
            for a in attrs:
                if isinstance(a, Space):
                    self.space = a
                elif isinstance(a, Loc):
                    self.location = a
                    self.space = a.space
                elif isinstance(a, SNCategory):
                    self.category = a
                elif isinstance(a, SNObject):
                    self.object = a
                elif isinstance(a, SNViewAngle):
                    self.view_angle = a
                elif isinstance(a, SNFixedObject):
                    self.view_angle = a
                else:
                    raise TypeError('Unknown type for attribute: ' +
                                    str(a) + ' ' + str(type(a)))

        self.when = when
        self.epoch = None
        self.deletable = deletable

    def __str__(self):
        return ' '.join([
            'Object:', 'location',
            str(self.location), 'category',
            str(self.category), 'object',
            str(self.object), 'view_angle',
            str(self.view_angle), 'when',
            str(self.when), 'epoch',
            str(self.epoch), 'deletable',
            str(self.deletable)
        ])

    def check_attrs(self):
        # sanity check
        return self.object.category == self.category and \
            self.view_angle.object == self.object

    def change_category(self, category: SNCategory):
        """
        change the category of the object, and resample the related attributes
        :param category: the new category
        """
        self.category = category
        self.object.category = category
        self.object.sample()
        self.view_angle.object = self.object
        self.view_angle.sample()

    def change_object(self, obj: SNObject):
        '''
        change the sn_object of the object
        :param obj:
        :return:
        '''
        self.category = obj.category
        self.object = obj
        self.view_angle.object = obj
        self.view_angle.sample()

    def change_view_angle(self, view_angle: SNViewAngle):
        self.category = view_angle.object.category
        self.object = view_angle.object
        self.view_angle = view_angle

    def change_fixed_object(self, fixed_object):
        self.category = fixed_object.object.category
        self.object = fixed_object.object
        self.view_angle = SNViewAngle(fixed_object.object, fixed_object.value)

    def change_attr(self, attr):
        if isinstance(attr, SNCategory):
            self.change_category(attr)
        elif isinstance(attr, SNObject):
            self.change_object(attr)
        elif isinstance(attr, SNViewAngle):
            self.change_view_angle(attr)
        elif isinstance(attr, SNFixedObject):
            self.change_fixed_object(attr)
        else:
            raise NotImplementedError()

    def compare_attrs(self, other, attrs: List[str] = None):
        assert isinstance(other, Object)

        if attrs is None:
            attrs = const.ATTRS
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def dump(self):
        """Returns representation of self suitable for dumping as json."""
        return {
            'location': str(self.location.value),
            'space': const.DATA.get_grid_key(self.location.space),
            'category': int(self.category.value),
            'object': int(self.object.value),
            'view angle': int(self.view_angle.value),
            'epochs': (self.epoch[0] if self.epoch[0] + 1 == self.epoch[1] else
                       list(range(*self.epoch))),
            'is_distractor': self.deletable
        }

    def to_static(self):
        """Convert self to a list of StaticObjects."""
        return [StaticObject(loc=self.location.value,
                             category=self.category.value,
                             object=self.object.value,
                             view_angle=self.view_angle.value,
                             epoch=epoch)
                for epoch in range(*self.epoch)]

    def merge(self, obj):
        """Attempt to merge with another object.

        Args:
          obj: an Object Instance

        Returns:
          bool: True if successfully merged, False otherwise
        """
        new_attr = dict()
        for attr_type in ['category', 'object', 'view_angle']:
            new_attr = getattr(obj, attr_type)
            self_attr = getattr(self, attr_type)
            if not self_attr.has_value and new_attr.has_value:
                self.change_attr(new_attr)
            elif new_attr.has_value and self_attr.has_value:
                return False
        return True

    def copy(self):
        """
        :return: deep copy of the object
        """
        new_obj = Object(attrs=[
            Loc(space=Space(self.location.space.value), value=self.location.value),
            Space(self.location.space.value),
            SNCategory(self.category.value),
            SNObject(self.category, self.object.value),
            SNViewAngle(self.object, self.view_angle.value)
        ])
        new_obj.when = self.when
        new_obj.epoch = self.epoch
        new_obj.deletable = self.deletable
        return new_obj


class ObjectSet(object):
    """A collection of objects."""

    def __init__(self, n_epoch, n_max_backtrack=4):
        """Initialize the collection of objects.

        Args:
          n_epoch: int, the number of epochs or frames in the object set
          n_max_backtrack: int or None
            If int, maximum number of epoch to look back when searching, at least 1.
            If None, will search the entire history
        """
        self.n_epoch = n_epoch
        self.n_max_backtrack = n_max_backtrack
        self.set = list()
        self.end_epoch = list()
        self.dict = defaultdict(list)  # key: epoch, value: list of obj

        self.last_added_obj = None  # Last added object
        self.loc = None

    def __iter__(self):
        return self.set.__iter__()

    def __str__(self):
        return '\n'.join([str(o) for o in self])

    def __len__(self):
        return len(self.set)

    def copy(self):
        """
        :return: deep copy of the Objset
        """
        objset_copy = ObjectSet(self.n_epoch, self.n_max_backtrack)
        objset_copy.set = {obj.copy() for obj in self.set}
        objset_copy.end_epoch = self.end_epoch.copy()
        objset_copy.dict = {epoch: [obj.copy() for obj in objs]
                            for epoch, objs in self.dict.items()}
        objset_copy.last_added_obj = self.last_added_obj.copy() if self.last_added_obj is not None else None
        return objset_copy

    def increase_epoch(self, new_n_epoch):
        """
        increase the number of epochs of the objset by initializing the new epoch indices with empty lists
        :param new_n_epoch: new number of epochs
        :return:
        """
        for i in range(self.n_epoch, new_n_epoch):
            self.dict[i] = list()
        self.n_epoch = new_n_epoch
        return

    def add(self,
            obj: Object,
            epoch_now,
            add_if_exist=False,
            delete_if_can=True,
            merge_idx=None,
            ):
        """Add an object at the current epoch

        This function will attempt to add the obj if possible.
        It will not only add the object to the objset, but also instantiate the
        attributes such as color, shape, and location if not already instantiated.

        Args:
          obj: an Object instance
          epoch_now: the current epoch when this object is added
          add_if_exist: if True, add object anyway. If False, do not add object if
            already exist
          delete_if_can: Boolean. If True, will delete object if it conflicts with
            current object to be added. Should be set to True for most situations.
          merge_idx: the absolute epoch for adding the object, used for merging task_info
            when lastk of different tasks results in ambiguous epoch_idx
        Returns:
          obj: the added object if object added. The existing object if not added.

        Raises:
          ValueError: if can't find place to put stimuli
        """

        if obj is None:
            return None

        # Check if already exists
        n_backtrack = self.n_max_backtrack
        obj_subset = self.select(
            epoch_now,
            space=obj.space,
            category=obj.category,
            object=obj.object,
            view_angle=obj.view_angle,
            when=obj.when,
            n_backtrack=n_backtrack,
            delete_if_can=delete_if_can,
            merge_idx=merge_idx
        )

        # True if more than zero objects match the attributes based on epoch_now and backtrack
        if obj_subset and not add_if_exist:
            self.last_added_obj = obj_subset[-1]
            return self.last_added_obj

        # instantiate the object attributes
        if not obj.location.has_value:
            avoid = [o.location.value for o in self.select_now(epoch_now)]
            obj.location = obj.space.sample(avoid=avoid)

        if obj.view_angle.has_value:
            obj.object = obj.view_angle.object
            obj.category = obj.view_angle.object.category
        else:
            if obj.object.has_value:
                obj.category = obj.object.category
            else:
                if not obj.category.has_value:
                    obj.category.sample()
                obj.object.category = obj.category
                obj.object.sample()
            obj.view_angle.object = obj.object
            obj.view_angle.sample()

        if obj.when is None:
            # If when is None, then object is always presented
            obj.epoch = [0, self.n_epoch]
        else:
            if merge_idx is None:
                try:
                    obj.epoch = [epoch_now - const.DATA.LASTMAP[obj.when], epoch_now - const.DATA.LASTMAP[obj.when] + 1]
                except:
                    raise NotImplementedError(
                        'When value: {:s} is not implemented'.format(str(obj.when)))
            else:
                obj.epoch = [merge_idx, merge_idx + 1]
        # Insert and maintain order
        i = bisect_left(self.end_epoch, obj.epoch[0])
        self.set.insert(i, obj)
        self.end_epoch.insert(i, obj.epoch[0])

        # Add to dict
        for epoch in range(obj.epoch[0], obj.epoch[1]):  ####xlei: didn't change above shifted here to avoid confusion
            self.dict[epoch].append(obj)
        self.last_added_obj = obj
        return self.last_added_obj

    def delete(self, obj):
        """Delete an object."""
        i = self.set.index(obj)
        self.set.pop(i)
        self.end_epoch.pop(i)

        for epoch in range(obj.epoch[0], obj.epoch[1]):
            self.dict[epoch].remove(obj)

    def select(self,
               epoch_now,
               space=None,
               category=None,
               object=None,
               view_angle=None,
               when=None,
               n_backtrack=None,
               delete_if_can=True,
               merge_idx=None
               ):
        """Select an object satisfying properties.

        Args:
          epoch_now: int, the current epoch
          space: None or a Loc instance, the location to be selected.
          color: None or a Color instance, the color to be selected.
          shape: None or a Shape instance, the shape to be selected.
          when: None or a string, the temporal window to be selected.
          n_backtrack: None or int, the number of epochs to backtrack
          delete_if_can: boolean, delete object found if can

        Returns:
          a list of Object instance that fit the pattern provided by arguments
        """
        space = space
        category = category or SNCategory(None)
        object = object or SNObject(category=category, value=None)
        view_angle = view_angle or SNViewAngle(sn_object=object, value=None)

        if not isinstance(category, SNCategory):
            raise TypeError('category has to be Category class, is instead of class ' +
                            str(type(category)))
        if not isinstance(object, SNObject):
            raise TypeError('object has to be Object class, is instead of class ' +
                            str(type(SNObject)))
        if not isinstance(view_angle, SNViewAngle):
            raise TypeError('view_angle has to be ViewAngle class, is instead of class ' +
                            str(type(SNViewAngle)))
        # assert isinstance(space, Space)

        if merge_idx is None:
            epoch_now -= const.DATA.LASTMAP[when]
        else:
            epoch_now = merge_idx

        return self.select_now(epoch_now, space, category, object, view_angle, delete_if_can)

    def select_now(self,
                   epoch_now,
                   space=None,
                   category=None,
                   object=None,
                   view_angle=None,
                   delete_if_can=False
                   ):
        """Select all objects presented now that satisfy properties."""
        # Select only objects that have happened
        subset = self.dict[epoch_now]

        if category is not None and category.has_value:
            subset = [o for o in subset if o.category == category]

        if object is not None and object.has_value:
            subset = [o for o in subset if o.object == object]

        if view_angle is not None and view_angle.has_value:
            subset = [o for o in subset if o.view_angle == view_angle]

        if space is not None and space.has_value:
            subset = [o for o in subset if space.include(o.location)]

        if delete_if_can:
            for o in subset:
                if o.deletable:
                    # delete obj from self
                    self.delete(o)
            # Keep the not-deleted
            subset = [o for o in subset if not o.deletable]

        # Order objects by location to have a deterministic ordering
        subset.sort(key=lambda o: (o.location.value, o.category.value, o.object.value, o.view_angle.value))

        return subset


def render_static_obj(canvas, obj, img_size):
    """Render a single object.

    Args:
      canvas: numpy array of type int8 (img_size, img_size, 3). Modified in place.
          Importantly, opencv default is (B, G, R) instead of (R,G,B)
      obj: StaticObject instance
      img_size: int, image size.
    """
    # Fixed specifications, see Space.sample()
    # when sampling, the most top-left position is (0.1, 0.1),
    # most bottom-right position is (0.9,0.9)
    # changing scaling requires changing space.sample)
    radius = int(0.25 * img_size)

    # Note that OpenCV color is (Blue, Green, Red)
    center = [0, 0]
    if obj.loc[0] < 0.5:
        center[0] = 56
    else:
        center[0] = 168
    if obj.loc[1] < 0.5:
        center[1] = 56
    else:
        center[1] = 168
    # center = (int(obj.location[0] * img_size), int(obj.location[1] * img_size))

    x_offset, x_end = center[0] - radius, center[0] + radius
    y_offset, y_end = center[1] - radius, center[1] + radius
    shape_net_obj = const.DATA.get_shapenet_object(obj, [radius * 2, radius * 2])
    assert shape_net_obj.size == (x_end - x_offset, y_end - y_offset)
    canvas[x_offset:x_end, y_offset:y_end] = shape_net_obj


def render_obj(canvas, obj, img_size):
    """Render a single object.

    Args:
      canvas: numpy array of type int8 (img_size, img_size, 3). Modified in place.
          Importantly, opencv default is (B, G, R) instead of (R,G,B)
      obj: Object or StaticObject instance, containing object information
      img_size: int, image size.
    """
    if isinstance(obj, StaticObject):
        render_static_obj(canvas, obj, img_size)
    else:
        render_static_obj(canvas, obj.to_static()[0], img_size)


def render(objsets, img_size=224):
    """Render a movie by epoch.

    Args:
      objsets: an ObjsetSet instance or a list of them
      img_size: int, size of image (both x and y)

    Returns:
      movie: numpy array (n_time, img_size, img_size, 3)
    """
    if not isinstance(objsets, list):
        objsets = [objsets]

    n_objset = len(objsets)
    n_epoch_max = max([objset.n_epoch for objset in objsets])

    # It's faster if use uint8 here, but later conversion to float32 seems slow
    movie = np.zeros((n_objset * n_epoch_max, img_size, img_size, 3), np.uint8)

    i_frame = 0
    for objset in objsets:
        for epoch_now in range(n_epoch_max):
            canvas = movie[i_frame:i_frame + 1, ...]  # return a view
            canvas = np.squeeze(canvas, axis=0)

            subset = objset.select_now(epoch_now)
            for obj in subset:
                render_obj(canvas, obj, img_size)
            i_frame += 1
    return movie


def add_fixation_cue(canvas, cue_size=0.05):
    """

    :param canvas: numpy array of shape: (img_size, img_size, 3)
    :param cue_size: size of the fixation cue
    :return:
    """
    img_size = canvas.shape[0]
    radius = int(cue_size * img_size)
    center = (canvas.shape[0] // 2, canvas.shape[1] // 2)
    thickness = int(0.02 * img_size)
    cv2.line(canvas, (center[0] - radius, center[1]),
             (center[0] + radius, center[1]), (255, 255, 255), thickness)
    cv2.line(canvas, (center[0], center[1] - radius),
             (center[0], center[1] + radius), (255, 255, 255), thickness)


def random_category():
    return SNCategory(random.choice(const.DATA.ALLCATEGORIES))


def another_category(category):
    all_categories = list(const.DATA.ALLCATEGORIES)
    try:
        all_categories.remove(category.value)
    except AttributeError:
        for c in category:
            all_categories.remove(c.value)
    return SNCategory(random.choice(all_categories))


def random_object(category):
    '''
    there are 14 objects in 12 categories
    :param category:
    :return: integer indicating the object id
    '''
    return SNObject(category=category, value=random.choice(const.DATA.ALLOBJECTS[category.value]))


def another_object(snObject):
    """
    select another object when there are constraints on the feature attributes
    """
    try:
        category = random_category()
        if category == snObject.category:
            all_objects = list(const.DATA.ALLOBJECTS[category.value])
            all_objects.remove(snObject.value)

            # resample category if no other objects in the same category
            if not all_objects:
                while category != snObject.category:
                    category = random_category()
                return random_object(category)
            return SNObject(category=category, value=random.choice(all_objects))
        else:
            return random_object(category)
    except AttributeError:
        category = random_category()
        all_objects = list(const.DATA.ALLOBJECTS[category.value])
        for o in snObject:
            all_objects.remove(o.value)
        return SNObject(category=category, value=random.choice(all_objects))


def random_view_angle(sn_object):
    return SNViewAngle(sn_object=sn_object,
                       value=random.choice(const.DATA.ALLVIEWANGLES[sn_object.category.value][sn_object.value]))


def another_view_angle(view_angle):
    try:
        obj = view_angle.object
        all_viewangles = list(const.DATA.ALLVIEWANGLES[obj.category.value][obj.value])
        all_viewangles.remove(view_angle.value)
        return SNViewAngle(sn_object=obj, value=random.choice(all_viewangles))
    except AttributeError:
        iterator = iter(view_angle)
        first_view = next(iterator)
        assert all(first_view.obj == view.obj for view in iterator)
        all_viewangles = list(const.DATA.ALLVIEWANGLES[first_view.category.value][first_view.value])
        for view in view_angle:
            all_viewangles.remove(view.value)
        return SNViewAngle(sn_object=first_view.obj,
                           value=random.choice(all_viewangles))


def random_fixed_object(sn_object):
    return SNFixedObject(sn_object=sn_object,
                         value=random.choice(const.DATA.ALLVIEWANGLES[sn_object.category.value][sn_object.value]))


def another_fixed_object(fixed_object):
    try:
        category = random_category()
        if category == fixed_object.object.category:
            all_objects = list(const.DATA.ALLOBJECTS[category.value])
            all_objects.remove(fixed_object.object.value)

            # resample category if no other objects in the same category
            if not all_objects:
                while category != fixed_object.object.category:
                    category = random_category()
                return random_fixed_object(random_object(category))
            obj = SNObject(category=category, value=random.choice(all_objects))
            return random_fixed_object(obj)
        else:
            return random_fixed_object(random_object(category))
    except AttributeError:
        raise NotImplementedError()


def random_attr(attr_type) -> Attribute:
    if attr_type == 'object':
        category = random_category()
        return random_object(category)
    elif attr_type == 'category':
        return random_category()
    elif attr_type == 'view_angle':
        obj = random_object(random_category())
        return random_view_angle(obj)
    elif attr_type == 'fixed_object':
        obj = random_object(random_category())
        return random_fixed_object(obj)
    elif attr_type == 'location':
        space = random_grid_space()
        return space.sample()
    else:
        raise NotImplementedError('Unknown attr_type :' + str(attr_type))
    # TODO: how to add new attributes from new datasets? function API?


def another_attr(attr):
    if isinstance(attr, SNCategory):
        return another_category(attr)
    elif isinstance(attr, SNObject):
        return another_object(attr)
    elif isinstance(attr, SNViewAngle):
        return another_view_angle(attr)
    elif isinstance(attr, SNFixedObject):
        return another_fixed_object(attr)
    elif isinstance(attr, Space):
        return another_loc(attr)
    elif isinstance(attr, Loc):
        return another_loc(attr)
    elif attr is const.DATA.INVALID:
        return attr
    else:
        raise TypeError(
            'Type {:s} of {:s} is not supported'.format(str(attr), str(type(attr))))


def random_loc(n=1):
    locs = list()
    for i in range(n):
        loc = random_attr('location')
        while loc in locs:
            loc = random_attr('location')
        locs.append(loc)
    return locs


def random_space():
    return random.choice(const.DATA.ALLSPACES)


def n_random_space():
    return len(const.DATA.ALLSPACES)


def random_when():
    """Random choose a when property.

    Here we use the numpy random generator to provide different probabilities.

    Returns:
      when: a string.
    """
    return np.random.choice(const.DATA.ALLWHENS, p=const.DATA.ALLWHENS_PROB)


def sample_when(n=1):
    """

    :param n:
    :return: a list of 'lastk', in random order
    """
    return np.random.choice(const.DATA.ALLWHENS, size=n, p=const.DATA.ALLWHENS_PROB, replace=False)


def check_whens(whens, existing_whens: list = None):
    # added check_whens to ensure 1 stimulus per frame
    existing_whens = set() if not existing_whens else set(existing_whens)
    len_ew = len(existing_whens)
    while len(set(whens) | existing_whens) != (len(whens)+len_ew):
        whens = sample_when(len(whens))
    return whens


def n_random_when():
    return len(const.DATA.ALLWHENS)


def sample_category(k):
    return [SNCategory(c) for c in random.sample(const.DATA.ALLCATEGORIES, k)]


def sample_object(k, category):
    return [SNObject(category=category, value=s)
            for s in random.sample(const.DATA.ALLOBJECTS[category.value], k)]


def sample_view_angle(k, obj: SNObject):
    return [SNViewAngle(sn_object=obj, value=v) for v in
            random.sample(const.DATA.ALLVIEWANGLES[obj.category.value][obj.value], k)]


def another_loc(loc):
    # to make things consistent with original COG code, only sample a different grid_space
    # another approach is keeping track of all grid_space in task class
    n_max_try = 100
    for i_try in range(n_max_try):
        grid_space = loc.space
        new_grid_space = another_grid_space(grid_space)
        new_loc = new_grid_space.sample()
        if not grid_space.include(new_loc):
            break
    return new_loc


def random_grid_space():
    return Space(random.choice(list(const.DATA.grid.values())))


def another_grid_space(space):
    keys = list(const.DATA.grid.keys()).copy()
    key = const.DATA.get_grid_key(space)
    keys.remove(key)
    new_key = random.choice(keys)
    return Space(const.DATA.grid[new_key])
