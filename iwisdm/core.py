"""
abstract base classes for any environment,
see class methods for their expected behaviours
"""

from collections import defaultdict
from typing import Tuple, Dict, List, Iterable, Any, Union

import networkx as nx
import numpy as np

from iwisdm.envs.registration import EnvSpec, StimData, Constant


class Attribute(object):
    """Base class for attributes."""

    env_spec: EnvSpec = None
    _stim_data: StimData = None
    constants: Constant = None

    def __init__(self, value):
        if isinstance(value, np.int_):
            value = int(value)
        elif isinstance(value, list):
            value = tuple(value)
        self.value = value
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
        info['value'] = self.value
        info.update(self.self_json())
        return info

    def sample(self):
        raise NotImplementedError('Abstract method')

    def resample(self, attr):
        raise NotImplementedError('Abstract method')

    def has_value(self):
        return self.value is not None


class Stimulus(object):
    """A stimulus on the screen.

    A stimulus is a collection of attributes.
    """
    env_spec: EnvSpec = None
    _stim_data: StimData = None
    constants: Constant = None
    epoch: List[int] = None

    def __str__(self):
        raise NotImplementedError

    def change_attr(self, attr: Attribute):
        raise NotImplementedError()

    def compare_attrs(self, other, attrs: List[str] = None):
        raise NotImplementedError

    def dump(self):
        """Returns representation of self suitable for dumping as json."""
        raise NotImplementedError

    def to_static(self):
        """Convert self to a list of StaticObjects."""
        raise NotImplementedError

    def merge(self, obj):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def render(self, *args):
        """
        return the rendered image of the stimulus
        """
        raise NotImplementedError


class StimuliSet(object):
    """A collection of objects."""
    env_spec: EnvSpec = None
    _stim_data: StimData = None
    constants: Constant = None

    def __init__(self, n_epoch):
        """Initialize the collection of objects.

        Args:
          n_epoch: int, the number of epochs or frames in the object set
        """
        self.n_epoch = n_epoch
        self.set = list()
        self.end_epoch = list()
        self.dict = defaultdict(list)  # key: epoch, value: list of obj

        self.last_added_obj = None  # Last added object
        self.location = None

    def __iter__(self):
        return self.set.__iter__()

    def __str__(self):
        return '\n'.join([str(o) for o in self])

    def __len__(self):
        return len(self.set)

    def copy(self):
        raise NotImplementedError

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
            stim: Stimulus,
            epoch_now,
            add_if_exist=False,
            delete_if_can=True,
            merge_idx=None,
            ):
        """Add an object at the current epoch

        This function will attempt to add the stim if possible.
        It will not only add the object to the objset, but also instantiate the
        attributes such as color, shape, and location if not already instantiated.

        Args:
          stim: a Stimulus instance
          epoch_now: the current epoch when this object is added
          add_if_exist: if True, add object anyway. If False, do not add object if
            already exist
          delete_if_can: Boolean. If True, will delete object if it conflicts with
            current object to be added. Should be set to True for most situations.
          merge_idx: the absolute epoch for adding the object, used for merging task_info
            when lastk of different tasks results in ambiguous epoch_idx
        Returns:
          stim: the added object if object added. The existing object if not added.

        Raises:
          ValueError: if can't find place to put stimuli
        """

        raise NotImplementedError

    def delete(self, stim):
        """Delete an object."""
        i = self.set.index(stim)
        self.set.pop(i)
        self.end_epoch.pop(i)

        for epoch in range(stim.epoch[0], stim.epoch[1]):
            self.dict[epoch].remove(stim)

    def select(self, *args):
        raise NotImplementedError

    def select_now(self, epoch_now: int, *args) -> Iterable[Stimulus]:
        """
        return a subset of the stimuli that are present at the current epoch,
        filter the stimuli based on the provided attributes
        """
        raise NotImplementedError


class Operator(object):
    """Base class for task constructors."""
    env_spec: EnvSpec = None
    _stim_data: StimData = None
    constants: Constant = None

    def __init__(self):
        # Whether this operator is the final operator
        self.child = list()
        self.parent = list()

    def __str__(self):
        pass

    def __hash__(self):
        if self.child:
            c_s = sorted(self.child, key=lambda o: o.__class__.__name__)
            return hash(tuple(
                [self.__class__.__name__] +
                [c for c in c_s if isinstance(c, Operator)]
            ))
        else:
            return hash(self.__class__.__name__)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.child) != len(other.child):
                return False
            else:
                for c, o_c in zip(
                        sorted(self.child, key=lambda o: o.__class__.__name__),
                        sorted(other.child, key=lambda o: o.__class__.__name__)
                ):
                    if isinstance(c, Operator) and isinstance(o_c, Operator):
                        if c != o_c:
                            return False
                return True
        return False

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

    def get_expected_input(self, *args, **kwargs):
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

    def get_children_targets(self, objset):
        """
        get the answers to each child recursively
        @param objset:
        @return:
        """
        if not isinstance(self, Operator):
            return
        target = str(self(objset, objset.n_epoch - 1))
        self_dict = dict()
        self_dict['question'] = str(self)
        self_dict['answer'] = target
        if not self.child:
            return self_dict
        self_dict['children'] = [
            c.get_children_targets(objset)
            for c in self.child if isinstance(c, Operator) and not isinstance(c, Attribute)
        ]
        return self_dict


class Task(object):
    """Base class for tasks."""
    env_spec: EnvSpec = None
    _stim_data: StimData = None
    constants: Constant = None

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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._operator == other._operator
        return False

    def __hash__(self):
        return hash(self._operator)

    def __str__(self):
        return str(self._operator)

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

    def topological_sort(self) -> List[Operator]:
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

    def guess_objset(self, *kwargs):
        """
        main function for generating frames based on task graph structure
        iterate through each node in topological order, and propagate the expected inputs from
        predecessors to the successors
        :return: the updated objset after going through the task graph
        """
        raise NotImplementedError

    @property
    def instance_size(self):
        """Return the total number of possible instantiated tasks."""
        raise NotImplementedError('instance_size is not defined for this task.')

    def get_target(self, objset):
        return [self(objset, objset.n_epoch - 1)]

    def get_subtask_targets(self, objset):
        """
        get the answers to each subtask
        @param objset:
        @return: nested lists of answers to each children operator
        """
        targets = [str(target) for target in self.get_target(objset)]
        self_dict = dict()
        self_dict['question'] = str(self)
        self_dict['answer'] = targets[0]
        if not self._operator.child:
            return self_dict
        self_dict['children'] = [
            c.get_children_targets(objset)
            for c in self._operator.child if isinstance(c, Operator)
        ]
        return self_dict

    def is_bool_output(self) -> bool:
        raise NotImplementedError


GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]
TASK = Tuple[Union[Operator, Attribute], Task]


class Env(object):
    """
    Base class for any environment.


    """
    metadata: Dict[str, Any] = dict()
    env_spec: EnvSpec = None
    task_gen_config: Dict[str, Any] = dict()
    constants: Constant = None
    stim_data: StimData = None

    @staticmethod
    def init_stim_data(dataset_fp: str, *args) -> StimData:
        """
        initialize the StimData for a specific environment
        @param dataset_fp:
        @param args:
        @return:
        """
        raise NotImplementedError

    @staticmethod
    def init_env_spec(*args):
        raise NotImplementedError

    def set_stim_data(self, stim_data: StimData):
        raise NotImplementedError

    def set_env_spec(self, env_spec: EnvSpec):
        raise NotImplementedError

    def generate_tasks(self, n: int = 1, *args) -> List[Tuple[GRAPH_TUPLE, TASK]]:
        """
        generate random tasks based on the environment specification
        @param n: how many tasks to generate
        @param args:
        @return: List of tasks, each task is a tuple of
        [
        1. task graph tuple [nx.DiGraph, operator count, root operator index],
        2. task tuple [Operator/Attribute, Task]
        ]
        """
        raise NotImplementedError

    def generate_trials(self,
                        tasks: List[Task],
                        mode: str,
                        **kwargs) -> List[Tuple[List[np.ndarray], List[Dict], Dict]]:
        """
        generate trials based on the tasks and which dataset split to sample stimuli from

        @param tasks: List of tasks to generate trials from
        @param mode: the dataset split [train/valid/test]
        @return: List of trials, each trial is a tuple of
        [List of images, Info dictionaries per task (in case of compositional tasks), Compositional task info dict]
        """
        raise NotImplementedError

    def render_trials(self):
        raise NotImplementedError
