"""
Code adapted from 'A Dataset and Architecture for Visual Reasoning with a Working Memory', Guangyu Robert Yang, et al.
Paper: https://arxiv.org/abs/1803.06092
Code: https://github.com/google/cog

Vocabulary of functional programs.

Contains the building blocks for permissible tasks.
"""

import json
from collections import defaultdict, OrderedDict
from typing import Tuple, Union, Dict, Callable, List

import networkx as nx
import numpy as np
import random

from cognitive import constants as const
from cognitive import stim_generator as sg


def obj_str(location=None, obj=None, category=None, view_angle=None,
            when=None):
    """Get a string describing an object with attributes."""

    location = location or sg.Location(space=sg.random_grid_space(), value=None)
    category = category or sg.SNCategory(None)
    obj = obj or sg.SNObject(category=category, value=None)
    view_angle = view_angle or sg.SNViewAngle(sn_object=obj, value=None)

    sentence = []
    if when is not None:
        sentence.append(when)
    if isinstance(category, sg.Attribute) and category.has_value():
        sentence.append(str(category))
    if isinstance(view_angle, sg.Attribute) and view_angle.has_value():
        sentence.append(str(view_angle))
    if isinstance(obj, sg.Attribute) and obj.has_value():
        sentence.append(str(obj))
    if isinstance(location, sg.Attribute) and location.has_value():
        sentence.append(str(location))
    else:
        sentence.append('object')

    if isinstance(category, Operator):
        sentence += ['with', str(category)]
    if isinstance(view_angle, Operator):
        sentence += ['with', str(view_angle)]
    if isinstance(location, Operator):
        sentence += ['with', str(location)]
    if isinstance(obj, Operator):
        sentence += ['with', str(category)]
    return ' '.join(sentence)


class Skip(object):
    """Skip this operator."""

    def __init__(self):
        pass


class Operator(object):
    """Base class for task constructors."""

    def __init__(self):
        # Whether or not this operator is the final operator
        self.child = list()
        self.parent = list()

    def __str__(self):
        pass

    def __call__(self, objset, epoch_now):
        del objset
        del epoch_now

    def copy(self):
        raise NotImplementedError

    def set_child(self, child):
        """Set operators as children."""
        try:
            child.parent.append(self)
            self.child.append(child)
        except AttributeError:
            for c in child:
                self.set_child(c)

    def get_expected_input(self, should_be=None):
        """Guess and update the objset at this epoch.

        Args:
          should_be: the supposed output
        """
        raise NotImplementedError('get_expected_input method is not defined.')

    def child_json(self):
        return [c.to_json() for c in self.child]

    def self_json(self):
        return {}

    def to_json(self):
        # return the dictionary for storing into json
        info = dict()
        info['name'] = self.__class__.__name__
        info['child'] = self.child_json()
        info.update(self.self_json())
        return info

    def check_attrs(self):
        """
        check if any children are operators
        :return: True if children contain no operators
        """
        for c in self.child:
            if isinstance(c, Operator):
                return False
        return True


class Select(Operator):
    """Selecting the objects that satisfy properties."""

    def __init__(self,
                 location=None,
                 category=None,
                 object=None,
                 view_angle=None,
                 attrs=None,
                 when=None,
                 space_type=None):
        super(Select, self).__init__()

        if attrs:
            for attr in attrs:
                if isinstance(attr, sg.Location):
                    location = attr
                elif isinstance(attr, sg.SNCategory):
                    category = attr
                elif isinstance(attr, sg.SNObject):
                    object = attr
                elif isinstance(attr, sg.SNViewAngle):
                    view_angle = attr
        location = location or sg.Location(space=sg.random_grid_space(), value=None)
        category = category or sg.SNCategory(None)
        object = object or sg.SNObject(category, None)
        view_angle = view_angle or sg.SNViewAngle(object, None)

        self.location, self.category, self.object, self.view_angle = location, category, object, view_angle
        self.set_child([location, category, object, view_angle])

        self.when = when
        self.space_type = space_type

    def __str__(self):
        return obj_str(
            self.location,
            self.category,
            self.object,
            self.view_angle,
            self.when
        )

    def __call__(self, objset, epoch_now):
        """Return subset of objset."""
        location = self.location(objset, epoch_now)
        category = self.category(objset, epoch_now)
        object = self.object(objset, epoch_now)
        view_angle = self.view_angle(object, epoch_now)

        if const.DATA.INVALID in (location, category, object, view_angle):
            return const.DATA.INVALID

        if self.space_type is not None:
            space = location.get_space_to(self.space_type)
        else:
            space = sg.Space(None)

        subset = objset.select(
            epoch_now,
            space=space,
            category=category,
            object=object,
            view_angle=view_angle,
            when=self.when,
        )
        return subset

    def copy(self):
        return Select(location=self.location, category=self.category, object=self.object, view_angle=self.view_angle,
                      when=self.when, space_type=self.space_type)

    def hard_update(self, obj: sg.Object):
        """
        change the attributes based on the attributes of the provided obj
        :param obj: an object on a frame that select needs to corresponds to
        :return: True if modification is successful
        """
        assert obj.check_attrs()

        self.category = obj.category
        self.object = obj.object
        self.view_angle = obj.view_angle
        self.location = obj.location
        return True

    def soft_update(self, obj: sg.Object):
        """
        change the attributes based on the attributes of the provided obj
        don't change the attribute if no value has been assigned

        :param obj: an object on a frame that select needs to corresponds to
        :return: True if modification is successful
        """
        assert obj.check_attrs()

        self.category = obj.category if self.category.has_value() else self.category
        self.object = obj.object if self.object.has_value() else self.object
        self.view_angle = obj.view_angle if self.view_angle.has_value() else self.view_angle
        self.location = obj.location if self.location.has_value() else self.location
        return True

    def get_expected_input(self, should_be, objset, epoch_now):
        """Guess objset for Select operator.

        Optionally modify the objset based on the target output (should_be),
        and pass down the supposed input attributes

        There are two sets of attributes to compute:
        (1) The attributes of the expected input, i.e. the attributes to select
        (2) The attributes of new object being added to the objset

        There are two main scenarios:
        (1) the target output is empty
        (2) the target output is not empty, then it has to be a list with a single
        Object instance in it

        When (1) the target output is empty, the goal is to make sure
        attr_newobject and attr_expectedinput differ by a single attribute

        When (2) the target output is not empty, then
        attr_newobject and attr_expectedinput are the same

        We first decide the attributes of the expected inputs.

        If target output is empty, then expected inputs is determined solely
        based on its actual inputs. If the actual inputs can already be computed,
        then use it, and skip the traversal of following nodes. Otherwise, use a
        random allowed value.

        If target output is not empty, again, if the actual inputs can be computed,
        use it. If the actual input can not be computed, but the attribute is
        specified by the target output, use that. Otherwise, use a random value.

        Then we determine the new object being added.

        If target output is empty, then randomly change one of the attributes that
        is not None.
        For self.space attribute, if it is an operator, then add the object at the
        opposite side of space

        If target output is not empty, use the attributes of the expected input

        Args:
          should_be: a list of Object instances
          objset: objset instance
          epoch_now: int, the current epoch

        Returns:
          objset: objset instance
          space: supposed input
          category: supposed input
          object: supposed input
          view_angle: supposed input

        Raises:
          TypeError: when should_be is not a list of Object instances
          NotImplementedError: when should_be is a list and has length > 1
        """

        if should_be is None:
            raise NotImplementedError('This should not happen.')

        if should_be:
            # Making sure should_be is a length-1 list of Object instance
            for s in should_be:
                if not isinstance(s, sg.Object):
                    raise TypeError('Wrong type for items in should_be: '
                                    + str(type(s)))

            if len(should_be) > 1:
                for s in should_be:
                    print(s)
                raise NotImplementedError()

            obj_should_be = should_be[0]

            attr_new_object = list()
            attr_type_not_fixed = list()

            # First evaluate the inputs of the operator that can be evaluated
            for attr_type in ['location', 'category', 'object', 'view_angle']:
                a = getattr(self, attr_type)
                attr = a(objset, epoch_now)
                # If the input is successfully evaluated
                if attr is not const.DATA.INVALID and attr.has_value():
                    if attr_type == 'location' and self.space_type is not None:
                        attr = attr.get_space_to(self.space_type)
                        print('space type is not none')
                    attr_new_object.append(attr)
                else:
                    attr_type_not_fixed.append(attr_type)

            # Add an object based on these attributes
            # Note that, objset.add() will implicitly select the objset
            obj: sg.Object = sg.Object(attr_new_object, when=self.when)
            obj = objset.add(obj, epoch_now, add_if_exist=False)

            if obj is None:
                return objset, Skip(), Skip(), Skip(), Skip()

            # If some attributes of the object are given by should_be, use them
            # change the attributes in obj based on should_be
            for attr_type in attr_type_not_fixed:
                a = getattr(obj_should_be, attr_type)
                if a.has_value():
                    if attr_type in ['category', 'object', 'view_angle']:
                        obj.change_attr(a)
                    else:
                        setattr(obj, attr_type, a)

            # If an attribute of select is an operator, then the expected input is the value of obj
            # attr_new_object is the output for the subsequent root children
            attr_expected_in = list()
            for attr_type in ['location', 'category', 'object', 'view_angle']:
                a = getattr(self, attr_type)
                if isinstance(a, Operator):
                    # Only operators need expected_in
                    # removed checking for location, since we're not flipping attributes
                    attr = getattr(obj, attr_type)
                    attr_expected_in.append(attr)
                else:
                    attr_expected_in.append(Skip())

        # no target output given, only happens when parent root is Exist, and Exist has no expected output
        # Basically if the target output is an empty set, then we place a different object.
        # We choose to place a different object here in order to prevent the network from solving some
        # tasks by simply counting the number of object. The object we
        # place differs from the object to be selected by only one attribute
        if not should_be:
            # First determine the attributes to flip later
            attr_type_to_flip = list()
            for attr_type in ['location', 'category', 'object', 'view_angle']:
                a = getattr(self, attr_type)
                # If attribute is operator or is a specified value
                if isinstance(a, Operator) or a.has_value():
                    attr_type_to_flip.append(attr_type)

            # Now generate expected input attributes
            attr_expected_in = list()
            attr_new_object = list()
            for attr_type in ['location', 'category', 'object', 'view_angle']:
                a = getattr(self, attr_type)
                attr = a(objset, epoch_now)
                if isinstance(a, Operator):
                    if attr is const.DATA.INVALID:
                        # Can not be evaluated yet, then randomly choose one
                        attr = sg.random_attr(attr_type)
                    attr_expected_in.append(attr)
                else:
                    attr_expected_in.append(Skip())

                if attr_type in attr_type_to_flip:
                    # Candidate attribute values for the new object
                    attr_new_object.append(attr)

            # Randomly pick one attribute to flip
            attr_type = random.choice(attr_type_to_flip)
            i = attr_type_to_flip.index(attr_type)
            if attr_type == 'location':
                # if the location of the object is determined by the location another object
                # e.g. Exist->Select->GetLoc->Select:
                # check if exist object with location of a specified category/object/view_angle

                # If flip location, place it in the opposite direction
                location = attr_new_object[i]
                attr_new_object[i] = location.get_opposite_space_to(self.space_type)
            else:
                # Select a different attribute
                attr_new_object[i] = sg.another_attr(attr_new_object[i])
                # Not flipping location, so place it in the correct direction
                if 'location' in attr_type_to_flip:
                    j = attr_type_to_flip.index('location')
                    location = attr_new_object[j]
                    attr_new_object[j] = location.get_space_to(self.space_type)

            # Add an object based on these attributes
            obj = sg.Object(attr_new_object, when=self.when)
            obj = objset.add(obj, epoch_now, add_if_exist=False)

        # Return the expected inputs
        return [objset] + attr_expected_in

    def self_json(self):
        return {'when': self.when}


class Get(Operator):
    """Get attribute of an object."""

    def __init__(self, attr_type, objs):
        """Get attribute of an object.

        Args:
          attr_type: string, color, shape, or location. The type of attribute to get.
          objs: Operator instance or Object instance
        """
        super(Get, self).__init__()
        self.attr_type = attr_type
        self.objs = objs
        assert isinstance(objs, Operator)
        self.set_child(objs)

    def __str__(self):
        words = [self.attr_type, 'of', str(self.objs)]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        """Get the attribute.

        By default, get the attribute of the unique object. If there are
        multiple objects, then return INVALID.

        Args:
          objset: objset
          epoch_now: epoch now

        Returns:
          attr: Attribute instance or INVALID
        """

        if isinstance(self.objs, Operator):
            objs = self.objs(objset, epoch_now)
        else:
            objs = self.objs

        if objs is const.DATA.INVALID:
            return const.DATA.INVALID
        elif len(objs) != 1:
            # Ambiguous or non-existent
            return const.DATA.INVALID
        else:
            attr = getattr(objs[0], self.attr_type)
            return attr

    def copy(self):
        new_objs = self.objs.copy()
        return Get(self.attr_type, new_objs)

    def get_expected_input(self, should_be):
        if should_be is None:
            should_be = sg.random_attr(self.attr_type)
        objs = sg.Object([should_be])
        return [objs]


class GetObject(Get):
    def __init__(self, objs):
        super(GetObject, self).__init__('object', objs)

    def __str__(self):
        words = ['identity', 'of', str(self.objs)]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)


class GetCategory(Get):
    def __init__(self, objs):
        super(GetCategory, self).__init__('category', objs)


class GetViewAngle(Get):
    def __init__(self, objs):
        super(GetViewAngle, self).__init__('view_angle', objs)


class GetLoc(Get):
    """Get location of object."""

    def __init__(self, objs):
        super(GetLoc, self).__init__('location', objs)


class Exist(Operator):
    """Check if object with property exists."""

    def __init__(self, objs):
        super(Exist, self).__init__()
        self.objs = objs
        assert isinstance(objs, Operator)
        self.set_child(objs)

    def __str__(self):
        words = [str(self.objs), 'exist']
        if not self.parent:
            words += ['?']
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        subset = self.objs(objset, epoch_now)
        if subset == const.DATA.INVALID:
            return const.DATA.INVALID
        elif subset:
            # If subset is not empty
            return True
        else:
            return False

    def copy(self):
        new_objs = self.objs.copy()
        return Exist(new_objs)

    def get_expected_input(self, should_be):

        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            should_be = [sg.Object()]
        else:
            should_be = []

        return should_be


class Switch(Operator):
    """Switch behaviors based on trueness of statement.

    Args:
      statement: boolean type operator
      do_if_true: operator performed if the evaluated statement is True
      do_if_false: operator performed if the evaluated statement is False
      invalid_as_false: When True, invalid statement is treated as False
      both_options_avail: When True, both do_if_true and do_if_false will be
        called the get_expected_input function
    """

    def __init__(self,
                 statement,
                 do_if_true,
                 do_if_false,
                 invalid_as_false=False,
                 both_options_avail=True):
        super(Switch, self).__init__()

        self.statement = statement
        self.do_if_true = do_if_true
        self.do_if_false = do_if_false

        self.set_child([statement, do_if_true, do_if_false])

        self.invalid_as_false = invalid_as_false
        self.both_options_avail = both_options_avail

    def __str__(self):

        words = [
            'if',
            str(self.statement) + ',',
            'then',
            str(self.do_if_true) + '?',
            'else',
            str(self.do_if_false)
        ]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        statement_true = self.statement(objset, epoch_now)
        if statement_true is const.DATA.INVALID:
            if self.invalid_as_false:
                statement_true = False
            else:
                return const.DATA.INVALID
        if statement_true:
            return self.do_if_true(objset, epoch_now)
        else:
            return self.do_if_false(objset, epoch_now)

    # def __getattr__(self, key):
    #     """Get attributes.

    #     The Switch operator should also have the shared attributes of do_if_true and
    #     do_if_false. Because regardless of the statement truthness, the common
    #     attributes will be true.

    #     Args:
    #       key: attribute

    #     Returns:
    #       attr1: common attribute of do_if_true and do_if_false

    #     Raises:
    #       ValueError: if the attribute is not common. Temporary.
    #     """
    #     attr1 = getattr(self.do_if_true, key)
    #     attr2 = getattr(self.do_if_false, key)
    #     if attr1 == attr2:
    #         return attr1
    #     else:
    #         raise ValueError()

    def copy(self):
        new_statement = self.statement.copy()
        new_do_if_true = self.do_if_true.copy()
        new_do_if_false = self.do_if_false.copy()
        return Switch(new_statement, new_do_if_true, new_do_if_false,
                      self.invalid_as_false, self.both_options_avail)

    def get_expected_input(self, should_be):
        """Guess objset for Switch operator.

        Here the objset is guessed based on whether or not the statement should be
        true.
        Both do_if_true and do_if_false should be executable regardless of the
        statement truthfulness. In this way, it doesn't provide additional
        information.

        Args:
          objset: objset
          epoch_now: current epoch

        Returns:
          objset: updated objset
        """
        if should_be is None:
            should_be = random.random() > 0.5

        return should_be, None, None


class IsSame(Operator):
    """Check if two attributes are the same."""

    def __init__(self, attr1, attr2):
        """Compare to attributes.

        Args:
          attr1: Instance of Attribute or Get
          attr2: Instance of Attribute or Get

        Raises:
          ValueError: when both attr1 and attr2 are instances of Attribute
        """
        super(IsSame, self).__init__()

        self.attr1 = attr1
        self.attr2 = attr2

        self.attr1_is_op = isinstance(self.attr1, Operator)
        self.attr2_is_op = isinstance(self.attr2, Operator)

        self.set_child([self.attr1, self.attr2])

        if (not self.attr1_is_op) and (not self.attr2_is_op):
            raise ValueError('attr1 and attr2 cannot both be Attribute instances.')

        self.attr_type = self.attr1.attr_type
        assert self.attr_type == self.attr2.attr_type

    def __str__(self):
        words = [self.attr1.__str__(), 'equals', self.attr2.__str__()]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        attr1 = self.attr1(objset, epoch_now)
        attr2 = self.attr2(objset, epoch_now)

        if (attr1 is const.DATA.INVALID) or (attr2 is const.DATA.INVALID):
            return const.DATA.INVALID
        else:
            return attr1 == attr2

    def copy(self):
        new_attr1 = self.attr1.copy()
        new_attr2 = self.attr2.copy()
        return IsSame(new_attr1, new_attr2)

    def get_expected_input(self, should_be, objset, epoch_now):
        if should_be is None:
            should_be = random.random() > 0.5
        # Determine which attribute should be fixed and which shouldn't

        attr1_value = self.attr1(objset, epoch_now)
        attr2_value = self.attr2(objset, epoch_now)

        attr1_fixed = attr1_value is not const.DATA.INVALID
        attr2_fixed = attr2_value is not const.DATA.INVALID

        if attr1_fixed:
            assert attr1_value.has_value()

        if attr1_fixed and attr2_fixed:
            # do nothing
            attr1_assign, attr2_assign = Skip(), Skip()
        elif attr1_fixed and not attr2_fixed:
            attr1_assign = Skip()
            attr2_assign = attr1_value if should_be else sg.another_attr(attr1_value)
        elif not attr1_fixed and attr2_fixed:
            attr1_assign = attr2_value if should_be else sg.another_attr(attr2_value)
            attr2_assign = Skip()
        else:
            if self.attr_type == 'view_angle':
                # if sample another view_angle if not should be
                # check for how many view_angles exist for the object first
                obj = sg.random_attr('object')
                while len(const.DATA.ALLVIEWANGLES[obj.category.value][obj.value]) < 2:
                    obj = sg.random_attr('object')
                attr = sg.random_view_angle(obj)
                attr1_assign = attr
                attr2_assign = attr if should_be else sg.another_attr(attr)
            elif self.attr_type == 'location':
                attr = sg.random_attr('location')
                attr1_assign = attr
                attr2_assign = attr.space.sample() if should_be else sg.another_attr(attr)
            else:
                attr = sg.random_attr(self.attr_type)
                attr1_assign = attr
                attr2_assign = attr if should_be else sg.another_attr(attr)
        return attr1_assign, attr2_assign


class NotSame(Operator):
    """Check if two attributes are not the same."""

    def __init__(self, attr1, attr2):
        """Compare to attributes.

        Args:
          attr1: Instance of Attribute or Get
          attr2: Instance of Attribute or Get

        Raises:
          ValueError: when both attr1 and attr2 are instances of Attribute
        """
        super(NotSame, self).__init__()

        self.attr1 = attr1
        self.attr2 = attr2

        self.attr1_is_op = isinstance(self.attr1, Operator)
        self.attr2_is_op = isinstance(self.attr2, Operator)

        self.set_child([self.attr1, self.attr2])

        if (not self.attr1_is_op) and (not self.attr2_is_op):
            raise ValueError('attr1 and attr2 cannot both be Attribute instances.')

        self.attr_type = self.attr1.attr_type
        assert self.attr_type == self.attr2.attr_type

    def __str__(self):
        words = [self.attr1.__str__(), 'not equals', self.attr2.__str__()]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        attr1 = self.attr1(objset, epoch_now)
        attr2 = self.attr2(objset, epoch_now)

        if (attr1 is const.DATA.INVALID) or (attr2 is const.DATA.INVALID):
            return const.DATA.INVALID
        else:
            return attr1 != attr2

    def copy(self):
        new_attr1 = self.attr1.copy()
        new_attr2 = self.attr2.copy()
        return NotSame(new_attr1, new_attr2)

    def get_expected_input(self, should_be, objset, epoch_now):
        if should_be is None:
            should_be = random.random() > 0.5
        # Determine which attribute should be fixed and which shouldn't
        attr1_value = self.attr1(objset, epoch_now)
        attr2_value = self.attr2(objset, epoch_now)

        attr1_fixed = attr1_value is not const.DATA.INVALID
        attr2_fixed = attr2_value is not const.DATA.INVALID

        if attr1_fixed:
            assert attr1_value.has_value()

        if attr1_fixed and attr2_fixed:
            # do nothing
            attr1_assign, attr2_assign = Skip(), Skip()
        elif attr1_fixed and not attr2_fixed:
            attr1_assign = Skip()
            attr2_assign = sg.another_attr(attr1_value) if should_be else attr1_value
        elif not attr1_fixed and attr2_fixed:
            attr1_assign = sg.another_attr(attr2_value) if should_be else attr2_value
            attr2_assign = Skip()
        else:
            if self.attr_type == 'view_angle':
                # if sample another view_angle if should be
                # check for how many view_angles exist for the object first
                obj = sg.random_attr('object')
                while len(const.DATA.ALLVIEWANGLES[obj.category.value][obj.value]) < 2:
                    obj = sg.random_attr('object')
                attr = sg.random_view_angle(obj)
                attr1_assign = attr
                attr2_assign = sg.another_attr(attr) if should_be else attr
            elif self.attr_type == 'location':
                attr = sg.random_attr('location')
                attr1_assign = attr
                attr2_assign = sg.another_attr(attr) if should_be else attr.space.sample()
            else:
                attr = sg.random_attr(self.attr_type)
                attr1_assign = attr
                attr2_assign = sg.another_attr(attr) if should_be else attr
        return attr1_assign, attr2_assign


class And(Operator):
    """And operator."""

    def __init__(self, op1, op2):
        super(And, self).__init__()
        self.op1, self.op2 = op1, op2
        self.set_child([op1, op2])

    def __str__(self):
        words = [str(self.op1), 'and', str(self.op2)]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        return self.op1(objset, epoch_now) and self.op2(objset, epoch_now)

    def copy(self):
        new_op1 = self.op1.copy()
        new_op2 = self.op2.copy()
        return And(new_op1, new_op2)

    def get_expected_input(self, should_be, objset, epoch_now):
        # should_be is the expected output
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            op1_assign, op2_assign = True, True
        else:
            r = random.random()
            # Assume the two operators are independent with P[op=True]=sqrt(0.5)
            # Here show the conditional probability given the output is False
            if r < 0.414:  # 2*sqrt(0.5) - 1
                op1_assign, op2_assign = False, True
            elif r < 0.828:
                op1_assign, op2_assign = True, False
            else:
                op1_assign, op2_assign = False, False

        return op1_assign, op2_assign


class Or(Operator):
    """Or operator."""

    def __init__(self, op1, op2):
        super(Or, self).__init__()
        self.op1, self.op2 = op1, op2
        self.set_child([op1, op2])

    def __str__(self):
        words = [str(self.op1), 'or', str(self.op2)]
        if not self.parent:
            words[-1] += '?'
        return ' '.join(words)

    def __call__(self, objset, epoch_now):
        return self.op1(objset, epoch_now) or self.op2(objset, epoch_now)

    def copy(self):
        new_op1 = self.op1.copy()
        new_op2 = self.op2.copy()
        return Or(new_op1, new_op2)

    def get_expected_input(self, should_be, objset, epoch_now):
        # should_be is the expected output
        if should_be is None:
            should_be = random.random() > 0.5

        if should_be:
            r = random.random()
            # Assume the two operators are independent with P[op=False]=1-sqrt(0.5)
            # Here show the conditional probability given the output is True
            if r < 0.414:
                op1_assign, op2_assign = False, True
            elif r < 0.828:
                op1_assign, op2_assign = True, False
            else:
                op1_assign, op2_assign = True, True
        else:
            op1_assign, op2_assign = False, False
        return op1_assign, op2_assign


class Task(object):
    """Base class for tasks."""

    def __init__(self, operator=None):
        if operator is None:
            self._operator = Operator()
        else:
            if not isinstance(operator, Operator):
                raise TypeError('operator is the wrong type ' + str(type(operator)))
            self._operator = operator

    def __call__(self, objset, epoch_now):
        # when you want to get the answer to the ask
        return self._operator(objset, epoch_now)

    def __str__(self):

        return str(self._operator)

    def _add_all_nodes(self, op: Union[Operator, sg.Attribute], visited: dict, G: nx.DiGraph, count: int):
        visited[op] = True
        parent = count
        node_label = type(op).__name__
        G.add_node(parent, label=node_label)
        if node_label == 'Switch':
            conditional_op, if_op, else_op = op.child[0], op.child[1], op.child[2]
            conditional_root = count + 1
            _, if_root = self._add_all_nodes(conditional_op, visited, G, conditional_root)
            _, else_root = self._add_all_nodes(if_op, visited, G, if_root + 1)
            _, else_node = self._add_all_nodes(else_op, visited, G, else_root + 1)
            G.add_edge(else_root, parent)
            G.add_edge(else_node, parent)
            G.add_edge(parent, conditional_root)
            return G, count
        else:
            for c in op.child:
                if isinstance(c, Operator) and not visited[c]:
                    child = count + 1
                    _, count = self._add_all_nodes(c, visited, G, child)
                    G.add_edge(parent, child)
                else:
                    if node_label != 'Select':
                        child = count + 1
                        G.add_node(child, label='CONST')
                        G.add_edge(parent, child)
                        count += 1
        return G, count

    def _get_all_nodes(self, op, visited):
        # used for topological sort, not need to read
        """Get the total number of operators in the graph starting with op."""
        visited[op] = True
        all_nodes = [op]
        for c in op.child:
            if isinstance(c, Operator) and not visited[c]:
                all_nodes.extend(self._get_all_nodes(c, visited))
        return all_nodes

    @property
    def _all_nodes(self):
        """Return all nodes in a list."""
        visited = defaultdict(lambda: False)
        return self._get_all_nodes(self._operator, visited)

    @property
    def operator_size(self):
        """Return the number of unique operators."""
        return len(self._all_nodes)

    def topological_sort_visit(self, node, visited, stack):
        """Recursive function that visits a root."""

        # Mark the current root as visited.
        visited[node] = True

        # Recur for all the vertices adjacent to this vertex
        for child in node.child:
            if isinstance(child, Operator) and not visited[child]:
                self.topological_sort_visit(child, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, node)

    def topological_sort(self):
        """Perform a topological sort."""
        nodes = self._all_nodes

        # Mark all the vertices as not visited
        visited = defaultdict(lambda: False)
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for node in nodes:
            if not visited[node]:
                self.topological_sort_visit(node, visited, stack)

        # Print contents of stack
        return stack

    def guess_objset(self, objset: sg.ObjectSet, epoch_now: int, should_be=None, temporal_switch=False):
        """
        main function for generating frames based on task graph structure
        iterate through each node in topological order, and propagate the expected inputs from
        predecessors to the successors
        :return: the updated objset after going through the task graph
        """
        nodes = self.topological_sort()
        should_be_dict = defaultdict(lambda: None)

        # if should_be is None, then the output is randomly sampled
        if should_be is not None:
            should_be_dict[nodes[0]] = should_be

        # iterate over all nodes in topological order
        # while updating the expected input from the successors/children of the current node
        for node in nodes:
            should_be = should_be_dict[node]
            # checking the type of operator
            if isinstance(should_be, Skip):
                inputs = Skip() if len(node.child) == 1 else [Skip()] * len(node.child)
            elif isinstance(node, Select):
                inputs = node.get_expected_input(should_be, objset, epoch_now)
                objset = inputs[0]
                inputs = inputs[1:]
            elif (isinstance(node, IsSame) or isinstance(node, NotSame)
                  or isinstance(node, And) or isinstance(node, Or)):
                inputs = node.get_expected_input(should_be, objset, epoch_now)
            else:
                inputs = node.get_expected_input(should_be)
            # e.g. node = IsSame, should_be = True,
            # expected_input is the output of the children operators

            # outputs is a list, if node is select, then get_expected_input adds object to objset
            if len(node.child) == 1:
                outputs = [inputs]  ## xl:for later interation
            else:
                outputs = inputs

            if isinstance(node, Switch):
                if temporal_switch:
                    children = node.child
                    if random.random() > 0.5:
                        children.pop(1)
                    else:
                        children.pop(2)
                else:
                    children = node.child
            else:
                children = node.child

            # Update the should_be dictionary for the children
            for c, output in zip(children, outputs):
                if not isinstance(c, Operator):  # if c is not an Operator
                    continue
                if isinstance(output, Skip):
                    should_be_dict[c] = Skip()
                    continue
                if should_be_dict[c] is None:
                    # If not assigned, assign
                    should_be_dict[c] = output
                # if child is an operator and there's already assigned expected output
                else:
                    # If assigned, for each object, try to merge them
                    # currently, only selects should have pre-assigned output
                    if isinstance(c, Select):
                        # Loop over new output
                        for o in output:
                            assert isinstance(o, sg.Object)
                            merged = False
                            # Loop over previously assigned outputs
                            for s in should_be_dict[c]:
                                # Try to merge
                                merged = s.merge(o)
                                if merged:
                                    break
                            if not merged:
                                should_be_dict[c].append(o)
                    else:
                        raise NotImplementedError()
        return objset

    @property
    def instance_size(self):
        """Return the total number of possible instantiated tasks."""
        raise NotImplementedError('instance_size is not defined for this task.')

    def get_target(self, objset):
        return [self(objset, objset.n_epoch - 1)]

    def is_bool_output(self):
        if self._operator in BOOL_OP:
            return True
        return False


class TemporalTask(Task):
    # goal: combine multiple tasks together
    # xlei: n_frames set to 6 todo: it does not work!
    def __init__(self, operator=None, n_frames=None, first_shareable=None, whens=None):
        super(TemporalTask, self).__init__(operator)
        self.n_frames = n_frames

        self._first_shareable = first_shareable
        self.whens = whens

    def copy(self):
        # duplicate the task
        new_task = TemporalTask()
        new_task.n_frames = self.n_frames
        new_task._first_shareable = self.first_shareable
        nodes = self.topological_sort()
        new_task._operator = self._operator.copy()
        return new_task

    @property
    def first_shareable(self):
        """

        :return: the frame at which the task is first shareable.
        if the task is non-shareable, first_shareable = len(task)
        if no input, start at random frame, including the possibility of non-shareable
        """
        if self._first_shareable is None:
            self._first_shareable = int(np.random.choice(np.arange(0, self.n_frames + 1)))
        return self._first_shareable

    @property
    def instance_size(self):
        # depending on the number of stimuli to sample from,
        # instance size determines the number of variations of the task
        pass

    @staticmethod
    def filter_selects(self, lastk=None) -> List[Select]:
        # filter for select operators that corresponds directly to an object and match lastk
        selects = list()

        for node in self.topological_sort():
            if isinstance(node, Select) and node.check_attrs():
                # print("what is node.when:", node.when)
                if lastk and node.when == lastk:
                    selects.append(node)
                else:
                    continue  # XLEI: only keep select operators that matches certain timestamp[lastk]
                    # selects.append(node)
        return selects

    def get_relevant_attribute(self, lastk):
        # return the attribute of the object that is not randomly selected on lastk frame
        # for merging check purpose
        attrs = set()

        for lastk_select in self.filter_selects(self, lastk):
            for parent_op in lastk_select.parent:
                if isinstance(parent_op, Get):
                    attrs.add(parent_op.attr_type)
                elif isinstance(parent_op, Exist):
                    for attr in const.ATTRS:
                        if getattr(lastk_select, attr).value:
                            attrs.add(attr)
                elif isinstance(parent_op, IsSame):
                    for op in parent_op.child:
                        if not isinstance(op, Operator):
                            attrs.add(op.attr_type)

        return attrs

    def reinit(self, copy, objs: List[sg.Object], lastk: str) -> List[sg.Object]:
        """update the task's selects in-place based on provided objects.

        :param lastk: the frame index with respect to the ending frame
        .g. if len(frames) is 3, last frame is last0, first frame is last2
        :param objs: list of objects that the selects need to match
        :type copy: copy of the TemporalTask
        :return: None if there are no leaf selects, list of objs otherwise
        """

        assert all([o.when == objs[0].when for o in objs])

        # find selects that match the lastk
        copy_filter_selects = copy.filter_selects(copy, lastk)
        filter_selects = self.filter_selects(self, lastk)

        # uncomment if multiple stim per frame
        # assert len(filter_selects) == len(copy_filter_selects)

        filter_objs = list()
        if filter_selects:
            if len(objs) < len(filter_selects):
                print('Not enough objects for select')
                return None
            # match objs on that frame with the number of selects
            filter_objs = random.sample(objs, k=len(filter_selects))
            # print("what is filter_objs:", filter_objs)
            for select, select_copy, obj in zip(filter_selects, copy_filter_selects, filter_objs):
                # each select that matches the lastk needs to change its attrs
                # to point to the reused object from existing frames
                select.soft_update(obj)
                select_copy.hard_update(obj)
        # print("after updating, what is filter_objs:", filter_objs)
        return filter_objs

    def generate_objset(self, *args, **kwargs):
        """Guess objset for all n_epoch. this is 1 trial

        Returns:
          objset: full objset for all n_epoch
        """

        n_epoch = self.n_frames
        const.DATA.MAX_MEMORY = n_epoch
        objset = sg.ObjectSet(n_epoch=n_epoch)

        # Guess objects
        # Importantly, we generate objset backward in time
        ## xlei: only update the last epoch
        epoch_now = n_epoch - 1

        objset = self.guess_objset(objset, epoch_now, *args, **kwargs)
        return objset

    def to_json(self, fp):
        info = dict()
        info['n_frames'] = int(self.n_frames)
        info['first_shareable'] = int(self.first_shareable)
        info['whens'] = self.whens
        info['operator'] = self._operator.to_json()

        with open(fp, 'w') as f:
            json.dump(info, f, indent=4)
        return info

    def to_graph(self):
        G = nx.DiGraph()
        visited = defaultdict(lambda: False)
        G, _ = self._add_all_nodes(self._operator, visited, G, 0)
        return G

    def draw_graph(self, fp, G: nx.DiGraph = None):
        if G is None:
            G = self.to_graph()
        G = G.reverse()
        A = nx.nx_agraph.to_agraph(G)
        A.draw(fp, prog='dot')
        return


TASK = Tuple[Union[Operator, sg.Attribute], TemporalTask]
GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def get_operator_dict() -> Dict[str, Operator]:
    # function to retrieve class names and their classes
    return {cls.__name__: cls for cls in all_subclasses(Operator)}


def get_attr_dict() -> Dict[str, Operator]:
    # function to retrieve class names and their classes
    return {cls.__name__: cls for cls in all_subclasses(sg.Attribute)}


def classify_operators(ops):
    counts = defaultdict(list)
    for i, op in enumerate(ops):
        if op == 'Select':
            counts['select'].append((i, op))
        elif op == 'Exist':
            counts['exist'].append((i, op))
        elif op == 'IsSame':
            counts['is_same'].append((i, op))
        elif op == 'NotSame':
            counts['not_same'].append((i, op))
        elif op == 'And' or op == 'Or' or op == 'Xor':
            counts['logic'].append((i, op))
        elif op == 'Switch':
            counts['switch'].append((i, op))
        elif 'Get' in op:
            counts['get'].append((i, op))
    return counts


def get_roots(G: nx.DiGraph):
    return [n for n, d in G.in_degree() if d == 0]


def get_leafs(G: nx.DiGraph):
    leafs = [x for x in G.nodes() if G.out_degree(x) == 0 and G.in_degree(x) == 1]
    assert all(leaf == 'Select' or leaf == 'CONST' for leaf in [G.nodes[node]['label'] for node in leafs])
    return leafs


def graph_to_operators(
        G: nx.DiGraph,
        root: int,
        operators: Dict[int, str],
        operator_families: Dict[str, Callable], whens) -> Union[Operator, sg.Attribute]:
    """
    given a task graph G, convert it into an operator that is already nested with its successors
    this is done by traversing the graph
    :param G: the task graph, nx.Graph
    :param root: the node number of the root operator
    :param operators: dictionary for converting node number to operator name
    :param operator_families: dict of class names and their classes
    :param whens: each select is associated with 1 when
    :return:
    """
    children = list(G.successors(root))
    if children:
        # check the root's type of operator
        if operators[root] == 'Select':
            attr_dict = {attr: None for attr in const.ATTRS}
            for child in children:
                # each child is an int indicating the node number
                if 'Get' in operators[child]:
                    attr = operators[child].split('Get')[1].lower()
                    attr = 'view_angle' if attr == 'viewangle' else attr
                    attr_dict[attr] = graph_to_operators(G, child, operators, operator_families, whens)
                else:
                    raise ValueError(f'Select cannot have {operators[child]} as a child operator')
            return Select(**attr_dict, when=whens[root])
        elif operators[root] == 'Exist':
            return Exist(graph_to_operators(G, children[0], operators, operator_families, whens))
        elif 'Get' in operators[root]:
            # init a Get operator
            return operator_families[operators[root]](
                graph_to_operators(G, children[0], operators, operator_families, whens))
        elif operators[root] in const.LOGIC_OPS:
            # init a boolean operator
            assert len(children) > 1
            ops = [graph_to_operators(G, c, operators, operator_families, whens) for c in children]
            if isinstance(ops[0], sg.Attribute) or isinstance(ops[1], sg.Attribute):
                if isinstance(ops[0], sg.Attribute):
                    attr = ops[0]
                    op = ops[1]
                    attr_i = 0
                else:
                    attr = ops[1]
                    op = ops[0]
                    attr_i = 1
                if attr.attr_type != op.attr_type:
                    ops[attr_i] = sg.random_attr(op.attr_type)
            return operator_families[operators[root]](ops[0], ops[1])
        else:
            raise ValueError(f"Unknown Operator {operators[root]}")
    else:
        if operators[root] == 'Select':
            return Select(when=whens[root])
        else:
            return sg.random_attr(random.choice(const.ATTRS))


def load_operator_json(
        op_dict: dict,
        operator_families: Dict[str, Callable] = None,
        attr_families: Dict[str, Callable] = None,
) -> Union[Operator, sg.Attribute]:
    """
    given json dictionary, convert it into an operator that is already nested with its successors
    :param op_dict: the operator dictionary
    :param operator_families:
    :param attr_families
    :return:
    """
    if operator_families is None:
        operator_families = get_operator_dict()
    if attr_families is None:
        attr_families = get_attr_dict()
    name = op_dict['name']
    if not 'value' in op_dict:
        children: List[dict] = op_dict['child']
        # check the type of operator
        if name == 'Select':
            attr_dict = {attr: None for attr in const.ATTRS}
            for d in children:
                if not 'value' in d:
                    if 'Get' in d['name']:
                        attr = d['name'].split('Get')[1].lower()
                        attr = 'view_angle' if attr == 'viewangle' else attr
                        attr_dict[attr] = load_operator_json(children[0], operator_families, attr_families)
                    else:
                        raise ValueError(f'Select cannot have {children[0]["name"]} as a child operator')
            return Select(**attr_dict, when=op_dict['when'])
        elif name == 'Exist':
            return Exist(load_operator_json(children[0], operator_families, attr_families))
        elif 'Get' in name:
            # init a Get operator
            return operator_families[name](
                load_operator_json(children[0], operator_families, attr_families))
        elif name in const.LOGIC_OPS:
            # init a boolean operator
            assert len(children) > 1
            ops = [load_operator_json(c, operator_families, attr_families) for c in children]
            if isinstance(ops[0], sg.Attribute) or isinstance(ops[1], sg.Attribute):
                if isinstance(ops[0], sg.Attribute):
                    attr = ops[0]
                    op = ops[1]
                    attr_i = 0
                else:
                    attr = ops[1]
                    op = ops[0]
                    attr_i = 1
                # match the attribute type of the operator
                if attr.attr_type != op.attr_type:
                    ops[attr_i] = sg.random_attr(op.attr_type)
            return operator_families[name](ops[0], ops[1])
        elif name == 'Switch':
            assert len(children) == 3
            statement, do_if_true, do_if_false = [load_operator_json(d, operator_families, attr_families) for d in
                                                  children]
            return Switch(statement, do_if_true, do_if_false)
        else:
            raise ValueError(f"Unknown Operator {name}")
    else:
        # we have reached an attribute
        init = {}
        if 'category' in op_dict:
            init['category'] = load_operator_json(op_dict['category'], operator_families, attr_families)
        elif 'sn_object' in op_dict:
            init['sn_object'] = load_operator_json(op_dict['sn_object'], operator_families, attr_families)

        return attr_families[name](value=op_dict['value'], **init)


def subtask_generation(subtask_graph: GRAPH_TUPLE, op_dict: dict = None, existing_whens: dict = None) \
        -> Tuple[TASK, dict]:
    existing_whens = dict() if not existing_whens else existing_whens
    # avoid duplicate whens across subtasks during switch generation
    subtask_G, root, _ = subtask_graph
    operator_families = get_operator_dict()

    if op_dict is None:
        op_dict = {node[0]: node[1]['label'] for node in subtask_G.nodes(data=True)}
    selects = [op for op in subtask_G.nodes() if 'Select' == op_dict[op]]

    # if max_mem = 7 (const.LASTMAP = [last6, ..., last0]), len(selects) = 4,
    # then there's no delay only if sample_when = [last3, last2, last1, last0]
    const.DATA.MAX_MEMORY = len(selects) + len(existing_whens.values())
    # resample whens so that every select is assigned to a different lastk
    whens = sg.check_whens(sg.sample_when(len(selects)), list(existing_whens.values()))
    n_frames = const.compare_when(whens) + 1  # find highest lastk to determine task number of frames

    whens = {select: when for select, when in zip(selects, whens)}
    existing_whens.update(whens)
    assert len(existing_whens.values()) == len(set(existing_whens.values())), 'whens are duplicated'
    op = graph_to_operators(subtask_G, root, op_dict, operator_families, whens)
    return (op, TemporalTask(operator=op, n_frames=n_frames, whens=whens)), existing_whens


def switch_generation(conditional: TASK, do_if: TASK, do_else: TASK, existing_whens: dict, **kwargs) -> TASK:
    """
    combines all 3 temporal tasks and initialize the switch operator
    """
    conditional_op, conditional_task = conditional
    if_op, if_task = do_if
    else_op, else_task = do_else

    op = Switch(conditional_op, if_op, else_op, **kwargs)
    n_frames = const.compare_when(existing_whens.values()) + 1

    # const.DATA.MAX_MEMORY = n_frames
    return op, TemporalTask(operator=op, n_frames=n_frames, whens=existing_whens)


get_family_dict = OrderedDict([
    ('category', GetCategory),
    ('object', GetObject),
    ('view_angle', GetViewAngle),
    ('location', GetLoc)
])

BOOL_OP = [IsSame, NotSame, Exist, And, Or]
