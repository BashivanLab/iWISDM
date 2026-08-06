""" procedural task generation.

based on the specified connectivity between operators,
construct random task graphs by adding operators layer by layer
also constructs tasks based on the generated graphs
"""

import random
from collections import defaultdict

from iwisdm.core import GRAPH_TUPLE, TASK  # noqa: F401  re-exported for env implementations
from iwisdm.envs.registration import EnvSpec


class _GlobalRandomProxy:
    """Picklable stand-in for the `random` module itself: forwards every
    attribute access to `random`, whose module-level functions are controlled
    by random.seed(). Unlike the module object, an instance of this class can
    be pickled (e.g. when the task generator ends up as multiprocessing.Pool
    task-dispatch state), because it carries no unpicklable state of its own.
    """

    def __getattr__(self, name):
        return getattr(random, name)


_GLOBAL_RANDOM = _GlobalRandomProxy()


class TaskGenerator:
    """
    base class for generating random task graphs and tasks
    based on the specified operator connectivity

    holds the operator-connectivity config and the complexity-budget helpers
    shared by all environments; environments implement generate_task
    """

    def __init__(self, env_spec: EnvSpec, rng: random.Random = None):
        """
        @param env_spec: environment specification with an auto_gen_config
        @param rng: source of randomness for graph-structure sampling;
        defaults to a picklable proxy for the random module, so that
        random.seed() keeps controlling generation as it does for the rest
        of the library, while self.rng itself stays safe to pickle (e.g.
        when this generator is dispatched through multiprocessing.Pool)
        """
        self.env_spec = env_spec
        self.rng = rng if rng is not None else _GLOBAL_RANDOM
        # dictionary specifying which operators can follow an operator,
        # e.g. In ShapeNet, GetCategory follows selecting an object
        # 4 operators are related to Select: category, location, view_angle, and the exact object
        # if the operator is None, then a random constant is sampled for that attribute
        self.config = env_spec.auto_gen_config

        self.op_dict = self.config['op_dict']
        self.root_ops = self.config['root_ops']
        self.boolean_ops = self.config['boolean_ops']
        op_dict = defaultdict(dict, **self.op_dict)
        # per-op minimal complexity: the depth/operator count needed to complete
        # the smallest valid subtask graph rooted at that op
        self.op_depth_limit = {k: v['min_depth'] for k, v in op_dict.items()}
        self.op_operators_limit = {k: v['min_op'] for k, v in op_dict.items()}

    def sample_root_helper(self, max_op: int, max_depth: int) -> str:
        """
        sample a root operator given constraints on the task complexity
        @param max_op: maximum number of operators allowed
        @param max_depth: maximum depth of the task graph
        @return: the name of a root operator that can complete a subtask graph
        """
        fits = [
            op for op in self.root_ops
            if self.op_depth_limit[op] <= max_depth and self.op_operators_limit[op] <= max_op
        ]
        if not fits:
            depth = {op: v for op, v in self.op_depth_limit.items() if op in self.root_ops}
            n_ops = {op: v for op, v in self.op_operators_limit.items() if op in self.root_ops}
            raise ValueError(f'The specified task complexity is too low given the available root operators\n'
                             f'The minimum depth of the specified root operators is: {depth}\n'
                             f'The minimum operator count of the specified root operators is: {n_ops}')
        return self.rng.choice(fits)

    def sample_children_helper(
            self,
            op_name: str,
            op_count: int,
            max_op: int,
            cur_depth: int,
            max_depth: int,
    ) -> str:
        """
        sample a child operator for op_name while keeping the task complexity bounded

        an op is a candidate if its minimal subtask graph still fits both the remaining
        depth and operator budgets; candidates that leave slack in the operator budget
        are preferred over exact fits. if no candidate fits, fall back to a minimum-depth
        op so the graph is completed as small as possible
        :param op_name: the current operator
        :param op_count: the current number of operators
        :param max_op: the maximum number of operators allowed
        :param cur_depth: the current depth
        :param max_depth: the maximum depth of the task graph
        :return: the name of a randomly sampled operator to follow the parent node
        """
        downstream_ops = self.op_dict[op_name]['downstream']
        if 'sample_dist' in self.op_dict[op_name]:  # sample_dist overrides complexity bounding
            return self.rng.choices(downstream_ops, weights=self.op_dict[op_name]['sample_dist'])[0]

        fits_depth = [op for op in downstream_ops if cur_depth + self.op_depth_limit[op] <= max_depth]
        # remaining operator budget after completing the op's minimal subtask graph
        op_slack = {
            op: max_op - (op_count + self.op_operators_limit[op])
            for op in fits_depth if op_count + self.op_operators_limit[op] <= max_op
        }
        if not op_slack:
            # the budgets cannot be met; complete the graph with a minimum-depth op
            min_depth = min(self.op_depth_limit[op] for op in downstream_ops)
            return self.rng.choice([op for op in downstream_ops if self.op_depth_limit[op] == min_depth])
        with_slack = [op for op, slack in op_slack.items() if slack > 0]
        return self.rng.choice(with_slack if with_slack else list(op_slack))

    def generate_task(self, *args):
        raise NotImplementedError
