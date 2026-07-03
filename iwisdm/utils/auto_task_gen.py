""" procedural task generation.

based on the specified connectivity between operators,
construct random task graphs by adding operators layer by layer
also constructs tasks based on the generated graphs
"""

from collections import defaultdict

from iwisdm.core import GRAPH_TUPLE, TASK  # noqa: F401  re-exported for env implementations
from iwisdm.envs.registration import EnvSpec


class TaskGenerator:
    """
    class for generating random task graphs and tasks based on the specified operator connectivity
    """

    def __init__(self, env_spec: EnvSpec):
        self.env_spec = env_spec
        # dictionary specifying which operators can follow an operator,
        # e.g. In ShapeNet, GetCategory follows selecting an object
        # 4 operators are related to Select: category, location, view_angle, and the exact object
        # if the operator is None, then a random constant is sampled for that attribute
        self.config = env_spec.auto_gen_config

        self.op_dict = self.config['op_dict']
        self.root_ops = self.config['root_ops']
        self.boolean_ops = self.config['boolean_ops']
        op_dict = defaultdict(dict, **self.op_dict)
        self.op_depth_limit = {k: v['min_depth'] for k, v in op_dict.items()}
        self.op_operators_limit = {k: v['min_op'] for k, v in op_dict.items()}

    def generate_task(self, *args):
        raise NotImplementedError


