"""
Code based on 'A Dataset and Architecture for Visual Reasoning with a Working Memory', Guangyu Robert Yang, et al.
Paper: https://arxiv.org/abs/1803.06092
Code: https://github.com/google/cog

High-level API for generating stimuli.

Objects are first generated abstractly, with high-level specifications
like location='random'.
Abstract relationships between objects can also be specified.

All objects and relationships are then collected into a ObjectSet.
The ObjectSet object can interpret the abstract specifications and instantiate
the stimuli in each trial.

Rendering function generates movies based on the instantiated stimuli
"""
from bisect import bisect_left
import random
from typing import List, Dict, Callable

from iwisdm.core import (
    Attribute,
    Stimulus,
    StimuliSet
)
from iwisdm.envs.shapenet.registration import SNEnvSpec, SNStimData
import iwisdm.envs.shapenet.registration as env_reg


class SNAttribute(Attribute):
    _stim_data: SNStimData = None
    env_spec: SNEnvSpec = None
    attr_type: str = None

    @property
    def stim_data(self):
        return self._stim_data

    @stim_data.setter
    def stim_data(self, value):
        self._stim_data = value

    def copy(self):
        """
        :return: deep copy of the attribute
        """
        return self.__class__(self.value)


class Space(SNAttribute):
    """Space class."""

    def __init__(self, value=None):
        super(Space, self).__init__(value)
        if self.value is None:
            self.value = [(0, 1), (0, 1)]
        else:
            self.value = value
        self.attr_type = 'space'

    def __hash__(self):
        return hash(tuple(self.value))

    def sample(self):
        return Space(random.choice(list(self.env_spec.grid.values())))

    def resample(self, attr):
        keys = list(self.env_spec.grid.keys()).copy()
        key = self.env_spec.get_grid_key(attr)
        keys.remove(key)
        new_key = random.choice(keys)
        return Space(self.env_spec.grid[new_key])

    def sample_loc(self, avoid=None):
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
            avoid += [mid_point]
        # avoid the mid-point for fixation cue

        n_max_try = 100
        avoid_radius2 = 0.04  # avoid radius squared

        dx = 0.001  # used to be 0.125 xuan => it does not matter now
        xrange = (self.value[0][0] + dx, self.value[0][1] - dx)
        dy = 0.001  # used to be 0.125 xuan => it does not matter now
        yrange = (self.value[1][0] + dy, self.value[1][1] - dy)

        not_overlapping = False
        loc_sample = None
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
        if not not_overlapping:
            raise RuntimeError('Could not sample another location')
        return Location(space=self, value=loc_sample)

    def include(self, loc):
        """Check if an unsampled location (a space) includes a location."""
        x, y = loc.value
        return ((self.value[0][0] < x < self.value[0][1]) and
                (self.value[1][0] < y < self.value[1][1]))

    def get_space_to(self, space_type: str):
        x0, x1 = self.value[0]
        y0, y1 = self.value[1]
        return _get_space_to(x0, x1, y0, y1, space_type)

    def get_opposite_space_to(self, space_type: str):
        """
        get the opposite space to the given space
        @param space_type:
        @return:
        """
        opposite_space = {
            'left': 'right',
            'right': 'left',
            'top': 'bottom',
            'bottom': 'top',
        }[space_type]
        return self.get_space_to(opposite_space)


class Location(SNAttribute):
    """Location class."""

    def __init__(self, space=None, value=None):
        """Initialize location.

        Args:
          value: None or a tuple of floats
          space: None or a tuple of tuple of floats
          If tuple of floats, then the actual
        """
        super(Location, self).__init__(value)
        if space is None:
            space = random_attr('space')
        self.attr_type = 'location'
        self.space = space

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.space == other.space
        return False

    def __hash__(self):
        return hash(self.space)

    def __str__(self):
        if len(self.env_spec.grid) == 4:
            key = self.env_spec.get_grid_key(self.space)
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

    def copy(self):
        """
        :return: deep copy of the location
        """
        return Location(space=Space(self.space.value), value=self.value)

    def sample(self):
        return self.space.sample_loc()

    def resample(self, attr):
        # sample a different grid_space, and a location in that space
        grid_space = attr.space
        new_grid_space = grid_space.resample(grid_space)
        new_loc = new_grid_space.sample_loc()
        if not grid_space.include(new_loc):
            return new_loc
        else:
            raise RuntimeError

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


class SNCategory(SNAttribute):
    def __init__(self, value):
        super(SNCategory, self).__init__(value)
        self.attr_type = 'category'

    def sample(self):
        return SNCategory(
            value=random.choice(self.stim_data.ALLCATEGORIES)
        )

    def resample(self, old_category):
        all_cats = self.stim_data.ALLCATEGORIES.copy()
        all_cats.remove(old_category.value)
        return SNCategory(
            value=random.choice(all_cats)
        )

    def __str__(self):
        if self.attr_type in self.stim_data.attr_with_mapping:
            return '' + self.stim_data.attr_with_mapping[self.attr_type][self.value]
        return 'category: ' + str(self.value)


class SNObject(SNAttribute):
    def __init__(self, category, value):
        super(SNObject, self).__init__(value)
        self.attr_type = 'object'
        self.category = category

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value and self.category == other.category
        return False

    def __hash__(self):
        return hash((self.category, self.value))

    def __str__(self):
        if self.attr_type in self.stim_data.attr_with_mapping:
            return '' + self.stim_data.attr_with_mapping[self.attr_type][self.value]
        return 'identity: ' + str(self.value)

    def copy(self):
        """
        :return: deep copy of the object
        """
        return SNObject(category=SNCategory(self.category.value), value=self.value)

    def sample(self, category=None):
        if category is None:
            category = random.choice(self.stim_data.ALLCATEGORIES)
        else:
            category = category.value
        obj = random.choice(self.stim_data.ALLOBJECTS[category])
        return SNObject(category=SNCategory(category), value=obj)

    def resample(self, old_obj):
        all_cats = self.stim_data.ALLCATEGORIES.copy()
        new_category = random.choice(all_cats)
        all_cats.remove(new_category)
        all_objects = list(self.stim_data.ALLOBJECTS[new_category])

        if new_category == old_obj.category.value:
            all_objects.remove(old_obj.value)

            # resample new category if no other objects in the same category
            if not all_objects:
                all_cats.remove(old_obj.category.value)
                new_category = random.choice(all_cats)
                all_objects = list(self.stim_data.ALLOBJECTS[new_category])
                return SNObject(category=SNCategory(new_category), value=random.choice(all_objects))
            return SNObject(category=SNCategory(new_category), value=random.choice(all_objects))
        else:
            return SNObject(category=SNCategory(new_category), value=random.choice(all_objects))

    def self_json(self):
        return {'category': self.category.to_json()}


class SNViewAngle(SNAttribute):
    def __init__(self, sn_object, value):
        super(SNViewAngle, self).__init__(value)
        self.attr_type = 'view_angle'
        self.object = sn_object

    def __str__(self):
        if self.attr_type in self.stim_data.attr_with_mapping:
            return '' + self.stim_data.attr_with_mapping[self.attr_type][self.value]
        return 'view_angle: ' + str(self.value)

    def copy(self):
        """
        :return: deep copy of the view angle
        """
        return SNViewAngle(
            sn_object=SNObject(
                category=SNCategory(self.object.category.value),
                value=self.object.value
            ),
            value=self.value
        )

    def sample(self, obj=None):
        if obj is None:
            category = random.choice(self.stim_data.ALLCATEGORIES)
            obj = random.choice(self.stim_data.ALLOBJECTS[category])
        else:
            category = obj.category.value
            obj = obj.value
        va = random.choice(self.stim_data.ALLVIEWANGLES[category][obj])
        return SNViewAngle(
            sn_object=SNObject(category=SNCategory(category), value=obj),
            value=va)

    def resample(self, old_va):
        obj = old_va.object
        all_vas = list(self.stim_data.ALLVIEWANGLES[obj.category.value][obj.value])
        all_vas.remove(old_va.value)
        return SNViewAngle(sn_object=obj, value=random.choice(all_vas))

    def self_json(self):
        return {'sn_object': self.object.to_json()}


class StaticObject(object):
    """Object that can be loaded from dataset and rendered."""

    def __init__(self, loc, category, object, view_angle, epoch):
        self.location = loc  # 2-tuple of floats
        self.category = category  # int
        self.object = object  # int
        self.view_angle = view_angle  # int
        self.epoch = epoch  # int


class SNStimulus(Stimulus):
    _stim_data: SNStimData = None
    env_spec: SNEnvSpec = None

    @property
    def stim_data(self):
        return self._stim_data

    @stim_data.setter
    def stim_data(self, value):
        self._stim_data = value

    def compare_attrs(self, other, attrs: List[str] = None):
        assert isinstance(other, SNStimulus)

        if attrs is None:
            attrs = env_reg.DATA.ATTRS
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class Object(SNStimulus):
    """A ShapeNet object on the screen.

    An object is a collection of attributes.

    Args:
      attrs: list [category, object, view_angle, location]
      when: string ('last', 'last1', 'last2')
      deletable: boolean. Whether this object is deletable. True if
        distractor.

    Raises:
      TypeError if location, category, object, view_angle
       are neither None nor respective Attributes
    """

    def __init__(self,
                 attrs=None,
                 when=None,
                 deletable=False):

        self.space = random_attr('space')
        self.location = Location(space=self.space)
        self.category = SNCategory(value=None)
        self.object = SNObject(self.category, value=None)
        self.view_angle = SNViewAngle(self.object, value=None)

        if attrs is not None:
            for a in attrs:
                if isinstance(a, Space):
                    self.space = a
                elif isinstance(a, Location):
                    self.location = a
                    self.space = a.space
                elif isinstance(a, SNCategory):
                    self.category = a
                elif isinstance(a, SNObject):
                    self.object = a
                elif isinstance(a, SNViewAngle):
                    self.view_angle = a
                else:
                    raise TypeError('Unknown type for attribute: ' +
                                    str(a) + ' ' + str(type(a)))

        self.when = when
        self.epoch = list()
        self.deletable = deletable

    def __str__(self):
        return ' '.join([
            'Object:',
            'location', str(self.location),
            'category', str(self.category),
            'object', str(self.object),
            'view_angle', str(self.view_angle),
            'when', str(self.when),
            'epoch', str(self.epoch),
            'deletable', str(self.deletable)
        ])

    def check_attrs(self):
        """
        check if the stimulus attributes are consistent
        @return: True if consistent, False otherwise
        """
        # sanity check
        return self.object.category == self.category and \
            self.view_angle.object == self.object

    def change_category(self, category: SNCategory):
        """
        change the category of the object, and resample the related attributes
        :param category: the new category
        """
        self.category = category
        self.object = self.object.sample(category=category)
        self.view_angle.object = self.object
        self.view_angle = self.view_angle.sample(obj=self.object)

    def change_object(self, obj: SNObject):
        """
        change the sn_object of the object
        :param obj:
        :return:
        """
        self.category = obj.category
        self.object = obj
        self.view_angle.object = obj
        self.view_angle = self.view_angle.sample(obj=obj)

    def change_view_angle(self, view_angle: SNViewAngle):
        self.category = view_angle.object.category
        self.object = view_angle.object
        self.view_angle = view_angle

    def change_attr(self, attr):
        if isinstance(attr, SNCategory):
            self.change_category(attr)
        elif isinstance(attr, SNObject):
            self.change_object(attr)
        elif isinstance(attr, SNViewAngle):
            self.change_view_angle(attr)
        else:
            raise NotImplementedError()

    def dump(self):
        """Returns representation of self suitable for dumping as json."""
        return {
            'location': str(self.location.value),
            'space': self.env_spec.get_grid_key(self.location.space),
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
            if not self_attr.has_value() and new_attr.has_value():
                self.change_attr(new_attr)
            elif new_attr.has_value() and self_attr.has_value():
                return False
        return True

    def copy(self):
        """
        :return: deep copy of the object
        """
        new_obj = Object(attrs=[
            Location(space=Space(self.location.space.value), value=self.location.value),
            Space(self.location.space.value),
            SNCategory(self.category.value),
            SNObject(self.category, self.object.value),
            SNViewAngle(self.object, self.view_angle.value)
        ])
        new_obj.when = self.when
        new_obj.epoch = self.epoch
        new_obj.deletable = self.deletable
        return new_obj


class ObjectSet(StimuliSet):
    """A collection of ShapeNet objects."""

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

        obj_subset = self.select(
            epoch_now,
            space=obj.space,
            category=obj.category,
            object=obj.object,
            view_angle=obj.view_angle,
            when=obj.when,
            delete_if_can=delete_if_can,
            merge_idx=merge_idx
        )

        # True if more than zero objects match the attributes based on epoch_now
        if obj_subset and not add_if_exist:
            self.last_added_obj = obj_subset[-1]
            return self.last_added_obj

        # instantiate the object attributes
        if not obj.location.has_value():
            avoid = [o.location.value for o in self.select_now(epoch_now)]
            obj.location = obj.space.sample_loc(avoid=avoid)

        if obj.view_angle.has_value():
            obj.object = obj.view_angle.object
            obj.category = obj.view_angle.object.category
        else:
            if obj.object.has_value():
                obj.category = obj.object.category
            else:
                if not obj.category.has_value():
                    obj.category = obj.category.sample()
                obj.object = obj.object.sample(category=obj.category)
            obj.view_angle = obj.view_angle.sample(obj=obj.object)

        if obj.when is None:
            # If when is None, then object is always presented
            obj.epoch = [0, self.n_epoch]
        else:
            if merge_idx is None:
                try:
                    obj.epoch = [epoch_now - env_reg.get_k(obj.when), epoch_now - env_reg.get_k(obj.when) + 1]
                except Exception:
                    raise NotImplementedError(
                        'When value: {:s} is not implemented'.format(str(obj.when)))
            else:
                obj.epoch = [merge_idx, merge_idx + 1]
        # Insert and maintain order
        i = bisect_left(self.end_epoch, obj.epoch[0])
        self.set.insert(i, obj)
        self.end_epoch.insert(i, obj.epoch[0])

        # Add to dict
        for epoch in range(obj.epoch[0], obj.epoch[1]):
            self.dict[epoch].append(obj)
        self.last_added_obj = obj
        return self.last_added_obj

    def select(
            self,
            epoch_now,
            space=None,
            category=None,
            object=None,
            view_angle=None,
            when=None,
            delete_if_can=True,
            merge_idx=None
    ):
        """Select an object satisfying properties.

        Args:
            epoch_now: int, the current epoch
            space: None or a Location instance, the location to be selected.
            category: None or an SNcategory instance, the ShapeNet category to be selected.
            object: None or an SNobject instance, the ShapeNet object to be selected.
            view_angle: None or an SNViewAngle instance, the ShapeNet view angle to be selected.

            when: None or a string, the temporal window to be selected.
            delete_if_can: boolean, delete object found if can
            merge_idx: the absolute epoch for adding the object, used for merging task_info
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
            epoch_now -= env_reg.get_k(when)
        else:
            epoch_now = merge_idx

        return self.select_now(epoch_now, space, category, object, view_angle, delete_if_can)

    def select_now(
            self,
            epoch_now,
            space=None,
            category=None,
            object=None,
            view_angle=None,
            delete_if_can=False
    ) -> List[Object]:
        """Select all objects presented now that satisfy properties.
        @param epoch_now: the current epoch
        @param space: the space to be selected
        @param category: the category to be selected
        @param object: the object identity to be selected
        @param view_angle: the view angle to be selected
        @param delete_if_can: boolean, delete distractors if specified to be True
        @return: a List of objects in the objset that satisfy the properties
        """
        # Select only objects that have happened
        subset = self.dict[epoch_now]

        if category is not None and category.has_value():
            subset = [o for o in subset if o.category == category]

        if object is not None and object.has_value():
            subset = [o for o in subset if o.object == object]

        if view_angle is not None and view_angle.has_value():
            subset = [o for o in subset if o.view_angle == view_angle]

        if space is not None and space.has_value():
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

    def copy(self):
        """
        :return: deep copy of the Objset
        """
        objset_copy = ObjectSet(self.n_epoch)
        objset_copy.set = {obj.copy() for obj in self.set}
        objset_copy.end_epoch = self.end_epoch.copy()
        objset_copy.dict = {epoch: [obj.copy() for obj in objs]
                            for epoch, objs in self.dict.items()}
        objset_copy.last_added_obj = self.last_added_obj.copy() if self.last_added_obj is not None else None
        return objset_copy


def _get_space_to(x0: float, x1: float, y0: float, y1: float, space_type: str) -> Space:
    """
    given the 2D coordinate of a stimulus and the relative space position,
    return a space that is to the left, right, top, or bottom of the stimulus
    :param x0: the first x coordinate of the stimulus location
    :param x1: the second x coordinate of the stimulus location
    :param y0: the first y coordinate of the stimulus location
    :param y1: the second y coordinate of the stimulus location
    :param space_type: the relative space position, ['left', 'right', 'top', 'bottom']
    :return: Space Tuple of Tuple of floats
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


def random_attr(attr_type: str) -> SNAttribute:
    """
    sample a random attribute
    @param attr_type: a ShapeNet attribute type,
    ['category', 'object', 'view_angle', 'location', 'space']
    @return: a random attribute of the specified type
    """
    if attr_type == 'category':
        tmp = SNCategory(None)
        return tmp.sample()
    elif attr_type == 'object':
        tmp = SNObject(None, None)
        return tmp.sample()
    elif attr_type == 'view_angle':
        tmp = SNViewAngle(None, None)
        return tmp.sample()
    elif attr_type == 'location':
        tmp = Space()
        space = tmp.sample()
        return space.sample_loc()
    elif attr_type == 'space':
        tmp = Space()
        return tmp.sample()
    else:
        raise NotImplementedError('Unknown attr_type :' + str(attr_type))


def another_attr(attr: Attribute) -> Attribute:
    """
    resample an attribute
    @param attr: the attribute class to be resampled
    @return: invalid or a resampled attribute
    """
    if attr == env_reg.DATA.INVALID:
        return attr
    else:
        return attr.resample(attr)


def get_attr_dict() -> Dict[str, Callable]:
    """
    retrieve attribute class names and their classes
    @return: dictionary of format {str: SNAttribute class}
    """
    # function to retrieve class names and their classes
    return {cls.__name__: cls for cls in env_reg.all_subclasses(SNAttribute)}
