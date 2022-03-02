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
like loc='random'.
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
import json
import os
from PIL import Image
import pandas as pd
import random
import numpy as np
import string

import cv2 as cv2
import tensorflow as tf

from cognitive import constants as const


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
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __ne__(self, other):
        """Define a non-equality test."""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return True

    def __hash__(self):
        return hash(self.value)

    # def __hash__(self):
    #     """Override the default hash behavior."""
    #     return hash(tuple(sorted(self.__dict__.items())))

    def resample(self):
        raise NotImplementedError('Abstract method.')

    @property
    def has_value(self):
        return self.value is not None


def _get_space_to(x0, x1, y0, y1, space_type):
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


class Loc(Attribute):
    """Location class."""

    def __init__(self, value=None):
        """Initialize location.

        Args:
          value: None or a tuple of floats
          space: None or a tuple of tuple of floats
          If tuple of floats, then the actual
        """
        super(Loc, self).__init__(value)
        self.attr_type = 'loc'

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
        if avoid is None:
            avoid = []

        n_max_try = 100
        avoid_radius2 = 0.04  # avoid radius squared
        dx = (self._value[0][1] - self._value[0][0]) * 0.125
        xrange = (self._value[0][0] + dx, self._value[0][1] - dx)
        dy = (self._value[1][1] - self._value[1][0]) * 0.125
        yrange = (self._value[1][0] + dy, self._value[1][1] - dy)
        for i_try in range(n_max_try):
            # Round to 3 decimal places to save space in json dump
            loc = (round(random.uniform(*xrange), 3),
                   round(random.uniform(*yrange), 3))

            not_overlapping = True
            for loc_avoid in avoid:
                not_overlapping *= ((loc[0] - loc_avoid[0]) ** 2 +
                                    (loc[1] - loc_avoid[1]) ** 2 > avoid_radius2)

            if not_overlapping:
                break

        return Loc(loc)

    def include(self, loc):
        """Check if an unsampled location (a space) includes a loc."""
        x, y = loc.value
        return ((self._value[0][0] < x < self._value[0][1]) and
                (self._value[1][0] < y < self._value[1][1]))

    def get_space_to(self, space_type):
        x0, x1 = self._value[0]
        y0, y1 = self._value[1]
        return _get_space_to(x0, x1, y0, y1, space_type)

    def get_opposite_space_to(self, space_type):
        opposite_space = {'left': 'right',
                          'right': 'left',
                          'top': 'bottom',
                          'bottom': 'top',
                          }[space_type]
        return self.get_space_to(opposite_space)


class SNObject(Attribute):
    def __init__(self, category, value):
        if value is not None:
            assert isinstance(category, SNCategory)
        super(SNObject, self).__init__(value)
        self.attr_type = 'object'
        self.category = category

    def sample(self):
        self.value = random_object(self.category).value

    def resample(self):
        self.value = another_object(self).value


class SNCategory(Attribute):
    def __init__(self, value):
        super(SNCategory, self).__init__(value)
        self.attr_type = 'category'

    def sample(self):
        self.value = random_category().value

    def resample(self):
        self.value = another_category(self).value


class SNViewAngle(Attribute):
    def __init__(self, value):
        super(SNViewAngle, self).__init__(value)
        self.attr_type = 'view_angle'

    def sample(self):
        self.value = random_view_angle().value

    def resample(self):
        self.value = another_view_angle(self).value


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
        subset.sort(key=lambda o: (o.loc, o.category, o.object, o.view_angle))
        return subset


class Object(object):
    """An object on the screen.

    An object is a collection of attributes.

    Args:
      loc: tuple (x, y)
      color: string ('red', 'green', 'blue', 'white')
      shape: string ('circle', 'square')
      when: string ('last', 'last1', 'last2',) ### todo: make it dynamic with free parameters
      deletable: boolean. Whether or not this object is deletable. True if
        distractors.

    Raises:
      TypeError if loc, color, shape are neither None nor respective Attributes
    """

    def __init__(self,
                 attrs=None,
                 when=None,
                 deletable=False):

        self.loc = Loc(value=None)
        self.space = Space(value=None)
        self.category = SNCategory(value=None)
        self.object = SNObject(self.category, value=None)
        self.view_angle = SNViewAngle(value=None)

        if attrs is not None:
            for a in attrs:
                if isinstance(a, Loc):
                    self.loc = a
                elif isinstance(a, Space):
                    self.space = a
                elif isinstance(a, SNObject):
                    self.object = a
                elif isinstance(a, SNCategory):
                    self.category = a
                elif isinstance(a, SNViewAngle):
                    self.view_angle = a
                else:
                    raise TypeError('Unknown type for attribute: ' +
                                    str(a) + ' ' + str(type(a)))

        self.when = when
        self.epoch = None
        self.deletable = deletable

    def __str__(self):
        return ' '.join([
            'Object:', 'loc',
            str(self.loc), 'category',
            str(self.category), 'object',
            str(self.object), 'view_angle',
            str(self.view_angle), 'when',
            str(self.when), 'epoch',
            str(self.epoch), 'deletable',
            str(self.deletable)
        ])

    def compare_attrs(self, other, attrs=None):
        assert isinstance(other, Object)

        if attrs is None:
            attrs = ['object', 'view_angle', 'category']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def dump(self):
        """Returns representation of self suitable for dumping as json."""
        return {
            'location': self.loc.value,
            'category': self.category.value,
            'object': self.object.value,
            'view angle': self.view_angle.value,
            'epochs': (self.epoch[0] if self.epoch[0] + 1 == self.epoch[1] else
                       list(range(*self.epoch))),
            'is_distractor': self.deletable
        }

    def to_static(self):
        """Convert self to a list of StaticObjects."""
        return [StaticObject(loc=self.loc.value,
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
        # TODO(gryang): What to do with self.when and self.loc?
        for attr_type in ['category', 'object', 'view_angle']:
            if not getattr(self, attr_type).has_value:
                new_attr[attr_type] = getattr(obj, attr_type)
            elif not getattr(obj, attr_type).has_value:
                new_attr[attr_type] = getattr(self, attr_type)
            else:
                return False

        for attr_type in ['category', 'object', 'view_angle']:
            setattr(self, attr_type, new_attr[attr_type])

        return True


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

    def copy(self):
        copy = ObjectSet(self.n_epoch, self.n_max_backtrack)
        copy.set = self.set.copy()
        copy.end_epoch = self.end_epoch.copy()
        copy.dict = self.dict.copy()
        copy.last_added_obj = self.last_added_obj
        return copy

    def __iter__(self):
        return self.set.__iter__()

    def __str__(self):
        return '\n'.join([str(o) for o in self])

    def __len__(self):
        return len(self.set)

    def increase_epoch(self, new_n_epoch):
        '''
        increase the number of epochs of the objset
        :param n_epoch: new number of epochs
        :return:
        '''
        for i in range(self.n_epoch, new_n_epoch):
            self.dict[i]
        self.n_epoch = new_n_epoch

    def add(self,
            obj: Object,
            epoch_now,
            add_if_exist=False,
            delete_if_can=True,
            merge_idx=None
            ):
        """Add an object at the current epoch

        This function will attempt to add the obj if possible.
        It will not only add the object to the objset, but also instantiate the
        attributes such as color, shape, and loc if not already instantiated.

        Args:
          obj: an Object instance
          epoch_now: the current epoch when this object is added
          add_if_exist: if True, add object anyway. If False, do not add object if
            already exist
          delete_if_can: Boolean. If True, will delete object if it conflicts with
            current object to be added. Should be set to True for most situations.

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
        )

        # True if more than zero objects match the attributes based on epoch_now and backtrack
        if obj_subset and not add_if_exist:
            self.last_added_obj = obj_subset[-1]
            return self.last_added_obj

        # instantiate the object attributes
        if not obj.loc.has_value:
            # Randomly generate locations, but avoid objects already placed now
            avoid = [o.loc.value for o in self.select_now(epoch_now)]
            obj.loc = obj.space.sample(avoid=avoid)

        if not obj.view_angle.has_value:
            obj.view_angle.sample()

        if not obj.object.has_value and not obj.category.has_value:
            if not obj.object.category.has_value:
                obj.category.sample()
                obj.object.category = obj.category
                obj.object.sample()
            else:
                obj.category = obj.object.category
                obj.object.sample()
        elif not obj.object.has_value and obj.category.has_value:
            obj.object.category = obj.category
            obj.object.sample()
        elif obj.object.has_value and not obj.category.has_value:
            obj.category = obj.object.category
        else:
            assert obj.category == obj.object.category

        if obj.when is None:
            # If when is None, then object is always presented
            obj.epoch = [0, self.n_epoch]
        else:
            if merge_idx is None:
                try:
                    obj.epoch = [epoch_now - const.LASTMAP[obj.when], epoch_now - const.LASTMAP[obj.when] + 1]
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

    def add_distractor(self, epoch_now):
        """Add a distractor."""
        category = random_category()
        object = random_object(category)
        obj1 = Object([category, object], when='last', deletable=True)
        self.add(obj1, epoch_now, add_if_exist=True)

    def delete(self, obj):
        """Delete an object."""
        i = self.set.index(obj)
        self.set.pop(i)
        self.end_epoch.pop(i)

        for epoch in range(obj.epoch[0], obj.epoch[1]):
            self.dict[epoch].remove(obj)

    def shift(self, x):
        """Shift every object in the set.

        Args:
          x: int, shift every object by x-epoch.
              An object that originally stays between (a, b) now stays between
              (max(0,a+x), b+x).

        Raises:
          ValueError: if n_epoch + x <= 0
        """
        self.n_epoch += x
        if self.n_epoch < 1:
            raise ValueError('n_epoch + x <= 0')

        new_set = list()
        new_end_epoch = list()
        new_dict = defaultdict(list)

        for obj in self.set:
            obj.epoch[0] = max((0, obj.epoch[0] + x))
            obj.epoch[1] += x
            if obj.epoch[1] > 0:
                new_set.append(obj)
                new_end_epoch.append(obj.epoch[1])

                for epoch in range(obj.epoch[0], obj.epoch[1]):
                    new_dict[epoch].append(obj)

        self.set = new_set
        self.end_epoch = new_end_epoch
        self.dict = new_dict

    def select(self,
               epoch_now,
               space=None,
               category=None,
               object=None,
               view_angle=None,
               when=None,
               n_backtrack=None,
               delete_if_can=True
               ):
        """Select an object satisfying properties.

        Args:
          epoch_now: int, the current epoch
          space: None or a Loc instance, the loc to be selected.
          color: None or a Color instance, the color to be selected.
          shape: None or a Shape instance, the shape to be selected.
          when: None or a string, the temporal window to be selected.
          n_backtrack: None or int, the number of epochs to backtrack
          delete_if_can: boolean, delete object found if can

        Returns:
          a list of Object instance that fit the pattern provided by arguments
        """
        space = space or Space(None)
        category = category or SNCategory(None)
        object = object or SNObject(category=category, value=None)
        view_angle = view_angle or SNViewAngle(None)

        if not isinstance(category, SNCategory):
            raise TypeError('category has to be Category class, is instead of class ' +
                            str(type(category)))
        if not isinstance(object, SNObject):
            raise TypeError('object has to be Object class, is instead of class ' +
                            str(type(SNObject)))
        if not isinstance(view_angle, SNViewAngle):
            raise TypeError('view_angle has to be ViewAngle class, is instead of class ' +
                            str(type(SNViewAngle)))
        assert isinstance(space, Space)

        epoch_now -= const.LASTMAP[when]

        # if n_backtrack is None:
        #   n_backtrack = self.n_max_backtrack   ### xlei: n_backtrack is deleted because of lastest does not exist anymore

        return self.select_now(epoch_now, space, category, object, view_angle, delete_if_can)
        # while epoch_now >= epoch_stop:
        #   subset = self.select_now(epoch_now, space, color, shape, delete_if_can)
        #   if subset:
        #     return subset
        #   epoch_now -= 1
        # return []

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
            subset = [o for o in subset if space.include(o.loc)]

        if delete_if_can:
            for o in subset:
                if o.deletable:
                    # delete obj from self
                    self.delete(o)
            # Keep the not-deleted
            subset = [o for o in subset if not o.deletable]

        # Order objects by location to have a deterministic ordering
        subset.sort(key=lambda o: (o.loc.value, o.category.value, o.object.value, o.view_angle.value))

        return subset


def get_shapenet_object(obj, obj_size):
    '''

    :param obj_size: a tuple of desired size
    :param category: the category of ShapeNet Object
    :return: a resized ShapeNet Object of obj_size
    '''
    shapnet_path = os.path.join(const.dir_path, 'min_shapenet_easy_angle')
    pickle_path = os.path.join(shapnet_path, 'train_min_shapenet_angle_easy_meta.pkl')
    images_path = os.path.join(shapnet_path, 'org_shapenet/train')

    df: pd.DataFrame = pd.read_pickle(pickle_path)
    obj_cat: pd.DataFrame = df.loc[(df['category'] == obj.category) &
                                   (df['object'] == obj.object) &
                                   (df['ang_mod'] == obj.view_angle)]
    assert len(obj_cat) > 0
    obj_ref = int(obj_cat.sample(1)['ref'])

    obj_path = os.path.join(images_path, f'{obj_ref}/image.png')
    img = Image.open(obj_path).convert('RGB').resize(obj_size)
    return img


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
    radius = int(0.125 * img_size)

    # Note that OpenCV color is (Blue, Green, Red)
    center = (int(obj.loc[0] * img_size), int(obj.loc[1] * img_size))

    x_offset, x_end = center[0] - radius, center[0] + radius
    y_offset, y_end = center[1] - radius, center[1] + radius
    shape_net_obj = get_shapenet_object(obj, [radius * 2, radius * 2])
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


def render_static(objlists, img_size=224, save_name=None):
    """Render a movie by epoch.

    Args:
      objlists: a list of lists of StaticObject instances
      img_size: int, size of image (both x and y)
      save_name: if not None, save movie at save_name

    Returns:
      movie: numpy array (n_time, img_size, img_size, 3)
    """

    n_epoch_max = max([o.epoch for objlist in objlists for o in objlist]) + 1

    # list of lists of lists. Each inner-most list contains
    # objects in a given epoch from a certain objlist.
    by_epoch = []
    key = lambda o: o.epoch
    for objects in objlists:
        by_epoch.append([])
        # Sort objects by epoch
        objects.sort(key=key)
        last_epoch = -1
        epoch_obj_dict = defaultdict(list,
                                     [(epoch, list(group)) for epoch, group
                                      in itertools.groupby(objects, key)])
        for i in range(n_epoch_max):
            # Order objects by location so that ordering is deterministic.
            # It controls occlusion.
            os = epoch_obj_dict[i]
            os.sort(key=lambda o: o.loc)
            by_epoch[-1].append(os)

    # It's faster if use uint8 here, but later conversion to float32 seems slow
    movie = np.zeros((len(objlists) * n_epoch_max, img_size, img_size, 3),
                     np.float32)

    i_frame = 0
    for objects in by_epoch:
        for epoch_objs in objects:
            canvas = movie[i_frame:i_frame + 1, ...]  # return a view
            canvas = np.squeeze(canvas, axis=0)
            for obj in epoch_objs:
                render_static_obj(canvas, obj, img_size)
            i_frame += 1
    assert i_frame == len(objlists) * n_epoch_max, '%d != %d' % (
        i_frame, len(objlists) * n_epoch_max)

    if save_name is not None:
        t_total = len(objlists) * n_epoch_max * 1.0  # need fps >= 1
        save_movie(movie, save_name, t_total)

    return movie


def render(objsets, img_size=224, save_name=None):
    """Render a movie by epoch.

    Args:
      objsets: an ObjsetSet instance or a list of them
      img_size: int, size of image (both x and y)
      save_name: if not None, save movie at save_name

    Returns:
      movie: numpy array (n_time, img_size, img_size, 3)
    """
    if not isinstance(objsets, list):
        objsets = [objsets]

    n_objset = len(objsets)
    n_epoch_max = max([objset.n_epoch for objset in objsets])

    # It's faster if use uint8 here, but later conversion to float32 seems slow
    movie = np.zeros((n_objset * n_epoch_max, img_size, img_size, 3), np.int8)

    i_frame = 0
    for objset in objsets:
        for epoch_now in range(n_epoch_max):
            canvas = movie[i_frame:i_frame + 1, ...]  # return a view
            canvas = np.squeeze(canvas, axis=0)

            subset = objset.select_now(epoch_now)
            for obj in subset:
                render_obj(canvas, obj, img_size)

            i_frame += 1

    if save_name is not None:
        t_total = n_objset * n_epoch_max * 1.0  # need fps >= 1
        save_movie(movie, save_name, t_total)

    return movie


def save_movie(movie, fname, t_total):
    """Save movie to file.

    Args:
      movie: numpy array (n_time, img_size, img_size, n_channels)
      fname: str, file name to be saved
      t_total: total time length of the video in unit second
    """
    print('Saving movie...')
    movie = movie.astype(np.uint8)
    # opencv interprets color channels as (B, G, R), so flip channel order
    movie = movie[..., ::-1]
    img_size = movie.shape[1]
    n_frame = len(movie)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    # filename, FOURCC (video code) (MJPG works), frame/second, framesize
    writer = cv2.VideoWriter(fname,
                             fourcc,
                             int(n_frame / t_total), (img_size, img_size))

    for frame in movie:
        writer.write(frame)
    writer.release()


def render_target(movie, target):
    """Specifically render the target response.

    Args:
      movie: numpy array (n_time, img_size, img_size, 3)
      target: list of tuples. List has to be length n_time

    Returns:
      movie_withtarget: same format as movie, but with target response shown

    Raises:
      TypeError: when target type is incorrect.
    """

    movie_withtarget = movie.copy()

    img_size = movie[0].shape[0]
    radius = int(0.02 * img_size)

    for frame, target_now in zip(movie_withtarget, target):
        if isinstance(target_now, Loc):
            loc = target_now.value
            center = (int(loc[0] * img_size), int(loc[1] * img_size))
            cv2.circle(frame, center, radius, (255, 255, 255), -1)
        else:
            if target_now is const.INVALID:
                string = 'invalid'
            elif isinstance(target_now, bool):
                string = 'true' if target_now else 'false'
            elif isinstance(target_now, Attribute):
                string = target_now.value
            elif isinstance(target_now, str):
                string = target_now
            else:
                raise TypeError('Unknown target type.')

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, string, (int(0.1 * img_size), int(0.8 * img_size)),
                        font, 0.5, (255, 255, 255))

    return movie_withtarget


def random_object(category):
    '''
    there are 14 objects in 12 categories
    :param category:
    :return: integer indicating the object id
    '''
    return SNObject(category=category, value=random.choice(const.ALLOBJECTS[category.value]))


def another_object(snObject):
    try:
        category = snObject.category
        all_objects = list(const.ALLOBJECTS[category.value])
        all_objects.remove(snObject.value)
        return SNObject(category=category, value=random.choice(all_objects))
    except AttributeError:
        iterator = iter(snObject)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        assert all(first.category == o.category for o in iterator)
        category = first.category
        all_objects = list(const.ALLOBJECTS[category.value])
        for o in snObject:
            all_objects.remove(o.value)
        return SNObject(category=category, value=random.choice(all_objects))


def random_category():
    return SNCategory(random.choice(const.ALLCATEGORIES))


def another_category(category):
    all_categories = list(const.ALLCATEGORIES)
    try:
        all_categories.remove(category.value)
    except AttributeError:
        for c in category:
            all_categories.remove(c.value)
    return SNCategory(random.choice(all_categories))


def random_view_angle():
    return SNViewAngle(random.choice(const.ALLVIEWANGLES))


def another_view_angle(view_angle):
    all_viewangles = list(const.ALLVIEWANGLES)
    try:
        all_viewangles.remove(view_angle.value)
    except AttributeError:
        for v in view_angle:
            all_viewangles.remove(v.value)
    return SNViewAngle(random.choice(all_viewangles))


def random_attr(attr_type, category=None):
    if attr_type == 'object':
        assert isinstance(category, SNCategory)
        return random_object(category)
    elif attr_type == 'category':
        return random_category()
    elif attr_type == 'view_angle':
        return random_view_angle()
    else:
        raise NotImplementedError('Unknown attr_type :' + str(attr_type))


def random_loc(n=1):
    locs = list()
    for i in range(n):
        loc = random_attr('loc')
        while loc in locs:
            loc = random_attr('loc')
        locs.append(loc)
    return locs


def random_space():
    return random.choice(const.ALLSPACES)


def n_random_space():
    return len(const.ALLSPACES)


def random_when(seed=None):
    """Random choose a when property.

    Here we use the numpy random generator to provide different probabilities.

    Returns:
      when: a string.
    """
    np.random.seed(seed=seed)
    return np.random.choice(const.ALLWHENS, p=const.ALLWHENS_PROB)


def sample_when(n=1, seed=None):
    return sorted([random_when(seed) for i in range(n)])


def n_random_when():
    return len(const.ALLWHENS)


def sample_category(k):
    return [SNCategory(c) for c in random.sample(const.ALLCATEGORIES, k)]


def sample_object(k, category):
    return [SNObject(category=category, value=s)
            for s in random.sample(const.ALLOBJECTS[category.value], k)]


def sample_view_angle(k):
    return [SNViewAngle(v) for v in random.sample(const.ALLVIEWANGLES, k)]


# def n_sample_shape(k):
#     return np.prod(range(len(const.ALLSHAPES) - k + 1, len(const.ALLSHAPES) + 1))
#
#
# def sample_colorshape(k):
#     return [
#         (Color(c), Shape(s)) for c, s in random.sample(const.ALLCOLORSHAPES, k)
#     ]
#
#
# def n_sample_colorshape(k):
#     return np.prod(
#         range(len(const.ALLCOLORSHAPES) - k + 1, len(const.ALLCOLORSHAPES) + 1))
#
#
# def another_color(color):
#     allcolors = list(const.ALLCOLORS)
#     try:
#         allcolors.remove(color.value)
#     except AttributeError:
#         for c in color:
#             allcolors.remove(c.value)
#
#     return Color(random.choice(allcolors))
#
#
# def another_shape(shape):
#     allshapes = list(const.ALLSHAPES)
#     try:
#         allshapes.remove(shape.value)
#     except AttributeError:
#         for s in shape:
#             allshapes.remove(s.value)
#     return Shape(random.choice(allshapes))


def another_loc(space):
    n_max_try = 100
    for i_try in range(n_max_try):
        loc = Loc((round(random.uniform(0.05, 0.95), 3),
                   round(random.uniform(0.05, 0.95), 3)))
        if not space.include(loc):
            break
    return loc


def another_attr(attr):
    if isinstance(attr, SNCategory):
        return another_category(attr)
    elif isinstance(attr, SNObject):
        return another_object(attr)
    elif isinstance(attr, SNViewAngle):
        return another_view_angle(attr)
    elif isinstance(attr, Space):
        return another_loc(attr)
    elif attr is const.INVALID:
        return attr
    else:
        raise TypeError(
            'Type {:s} of {:s} is not supported'.format(str(attr), str(type(attr))))


def main(argv):
    del argv  # Unused.


if __name__ == '__main__':
    tf.app.run(main)
