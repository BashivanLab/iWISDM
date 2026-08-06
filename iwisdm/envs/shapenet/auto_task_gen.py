""" procedural task generation.

based on the specified connectivity between operators,
construct random task graphs by adding operators layer by layer
also constructs tasks based on the generated graphs
"""

import random
from typing import Tuple, List

import networkx as nx

from iwisdm.core import Operator, GRAPH_TUPLE, TASK
from iwisdm.utils.auto_task_gen import TaskGenerator
import iwisdm.envs.shapenet.task_generator as tg
from iwisdm.envs.shapenet.registration import SNEnvSpec
from iwisdm.utils.helper import compute_get_signature


class _GraphBuilder:
    """
    accumulates the nodes and edges of one subtask graph while the generator
    samples operators under the operator/depth budgets

    holding the graph, the node counter and the budgets here avoids threading
    them through every level of the recursion
    """

    def __init__(self, gen: 'SNTaskGenerator', count: int, max_op: int, max_depth: int):
        self.gen = gen
        self.G = nx.DiGraph()
        self.op_count = count
        self.max_op = max_op
        self.max_depth = max_depth

    def grow(self, op_name: str, cur_depth: int, select_op=False, select_downstream=None):
        """
        complete the subtask graph below the op_name node,
        which has just been allocated with node number self.op_count
        @param op_name: the operator whose children are grown
        @param cur_depth: depth of the op_name node
        @param select_op: Boolean, whether a Select may be followed by a Get operator
        @param select_downstream: the downstream options Selects can sample from
        """
        parent = self.op_count
        # Exist operator always follows a select with operator as child
        select_op = True if op_name == 'Exist' else select_op

        children = self.gen.sample_children_op(
            op_name=op_name,
            op_count=self.op_count + 1,
            max_op=self.max_op,
            cur_depth=cur_depth + 1,
            max_depth=self.max_depth,
            select_op=select_op,
            select_downstream=select_downstream,
        )
        cur_depth += 1

        if op_name == 'Select' and any('Get' in c for c in children):
            # stop generating overly complex tasks where selects with operator children
            # follow more operators
            select_downstream = ['None'] * self.gen.N_SELECT_SLOTS
            select_op = False

        children = self.gen.maybe_swap_in_const(op_name, children)
        for op in children:
            if op == 'None':  # no node is added for empty slots
                continue
            child = self.op_count = self.op_count + 1
            if op != 'CONST':  # CONST is a leaf; anything else grows a sub-branch
                self.grow(op, cur_depth, select_op, select_downstream)
            self.G.add_node(child, label=op)
            self.G.add_edge(parent, child)


class SNTaskGenerator(TaskGenerator):
    """
    class for generating random task graphs and tasks based on the specified operator connectivity
    """

    # probability that a Select child slot stays empty ('None') rather than
    # getting a Get operator
    SELECT_NONE_PROB = 0.8
    # number of child slots under a Select: category, object, view_angle, location
    N_SELECT_SLOTS = 4
    # probability that the pre-existing subtask becomes the do_if branch of a switch
    SWITCH_IF_PROB = 0.5

    def __init__(self, env_spec: SNEnvSpec, rng: random.Random = None):
        """
        constructor for the SNTaskGenerator
        @param env_spec: the task generation configuration,
        see iwisdm/envs/shapenet/registration.py for default env_spec
        @param rng: source of randomness for graph-structure sampling;
        defaults to the random module itself, so that random.seed() keeps
        controlling generation as it does for the rest of the library
        """
        super().__init__(env_spec, rng)
        self.max_op = self.config['max_op']
        self.max_depth = self.config['max_depth']
        self.select_limit = self.config['select_limit']
        self.max_switch = self.config['max_switch']
        self.switch_threshold = self.config['switch_threshold']
        # constant-comparison settings are optional in custom configs;
        # by default no Get child is swapped for a constant
        self.compare_const_prob = self.config.get('compare_const_prob', 0.0)
        self.const_parent_ops = self.config.get('const_parent_ops', [])
        self.indexable_get_ops = self.config.get('indexable_get_ops', [])

    @staticmethod
    def count_depth_and_op(op: Operator) -> Tuple[int, int]:
        """
        count the number of operators and the depth of the operator graph
        @param op: the root operator with nested successors
        @return: [node count, depth count]
        """
        op_count, depth_count = 0, 0
        if not isinstance(op, Operator):
            return op_count, depth_count
        if op.child:
            # iterate over the nodes, and increment count
            depth_counts = list()
            for c in op.child:
                add_count = SNTaskGenerator.count_depth_and_op(c)
                op_count += add_count[0]
                depth_counts += [add_count[1]]
            depth_count += max(depth_counts)
        else:
            # at the leaf nodes
            return 1, 1
        op_count += 1
        depth_count += 1
        return op_count, depth_count

    @staticmethod
    def switch_generator(conditional: GRAPH_TUPLE, do_if: GRAPH_TUPLE, do_else: GRAPH_TUPLE) -> GRAPH_TUPLE:
        """
        function to generate a switch operator based on the conditional, do_if, and do_else subtask graphs
        stitches the 3 subtask graphs together to form a switch task graph
        @param conditional:
        @param do_if:
        @param do_else:
        @return:
        """
        # combine the 3 subtasks graphs into the switch task graph by using networkx compose_all
        do_if_graph, do_if_root, do_if_node = do_if
        do_else_graph, do_else_root, do_else_node = do_else
        conditional_graph, conditional_root, conditional_node = conditional

        # combine all 3 graphs and add edges
        G: nx.DiGraph = nx.compose_all([do_if_graph, do_else_graph, conditional_graph])
        switch_count = conditional_node + 1
        G.add_node(switch_count, label='Switch')
        # note that all directed edges are reversed later at task construction
        # after reversal, switch operator is connected to the selects of do_if and do_else
        # the root of conditional is then connected to the switch operator
        G.add_edge(do_if_node, switch_count)
        G.add_edge(do_else_node, switch_count)
        G.add_edge(switch_count, conditional_root)
        return G, switch_count, switch_count

    def maybe_swap_in_const(self, op_name: str, children: List[str]) -> List[str]:
        """
        for comparison operators, possibly replace one child with a constant

        the current environment only allows comparing category and location constants;
        view_angle and object are not compared to constants
        (remove this check if there's a mapping for object/view_angle idx)
        @param op_name: the parent operator
        @param children: the sampled children operators
        @return: the children, with at most one replaced by 'CONST'
        """
        if op_name not in self.const_parent_ops:
            return children

        if not any(op not in self.indexable_get_ops for op in children):
            if self.rng.random() < self.compare_const_prob:
                idx = self.rng.randrange(len(children))
                children[idx] = 'CONST'
        elif self.compare_const_prob == 1.0:
            # force compare to CONST by changing children to indexable Get operators
            if self.op_dict[op_name]['same_children_op']:
                n_downstream = self.op_dict[op_name]['n_downstream']
                child = self.rng.choice(self.indexable_get_ops)
                children = [child for _ in range(n_downstream)]
            else:
                children = [
                    self.rng.choice(self.indexable_get_ops)
                    if node not in self.indexable_get_ops else node
                    for node in children
                ]
            idx = self.rng.randrange(len(children))
            children[idx] = 'CONST'
        return children

    def sample_children_op(
            self,
            op_name: str,
            op_count: int,
            max_op: int,
            cur_depth: int,
            max_depth: int,
            select_op: bool,
            select_downstream: List[str]
    ) -> List[str]:
        """
        sample the children operators given the operator name
        :param op_name: the name of the parent node
        :param op_count: operator count
        :param max_op: max number of operators allowed
        :param cur_depth: current depth
        :param max_depth: max depth allowed for the task graph
        :param select_op: Boolean, does select follow an operator
        :param select_downstream: the downstream options that can be sampled from
        :return: list of operators
        """
        n_downstream = self.op_dict[op_name]['n_downstream']  # how many children need to be sampled
        if n_downstream == 1:
            return [self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)]

        if op_name == 'Select':
            if cur_depth + 2 > max_depth or op_count + 2 > max_op:
                return ['None'] * n_downstream
            if not select_op:  # Select not followed by a Get operator: all slots constant
                return ['None'] * n_downstream

            if select_downstream is None:
                # copy so sampling never mutates the config's downstream list
                select_downstream = list(self.op_dict['Select']['downstream'])
            children = list()
            op_sampled = False
            for _ in range(n_downstream):
                # make sure not too many operators follow select
                if self.rng.random() < self.SELECT_NONE_PROB:
                    children.append('None')
                elif select_downstream and not op_sampled:  # if the list is not empty, sample a downstream op
                    get = self.rng.choice(select_downstream)
                    children.append(get)
                    op_sampled = True
                    select_downstream.remove(get)
                else:
                    children.append('None')
            return children

        if self.op_dict[op_name]['same_children_op']:
            child = self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)
            return [child for _ in range(n_downstream)]
        return [
            self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)
            for _ in range(n_downstream)
        ]

    def subtask_graph_generator(
            self,
            count=0,
            max_op=20,
            max_depth=10,
            select_limit=False,
            root_op=None,
    ) -> GRAPH_TUPLE:
        """
        function for generating subtask graphs
        uses networkx to compose the task graphs
        @param count: the node number of the graph root
        @param max_op: the maximum number of operators allowed in the task graph
        @param max_depth: the maximum depth of the task graph
        @param select_limit: whether to add operator after selects. if True, then constants are sampled.
        If false, then operators could be sampled
        @param root_op: the root operator, sampled from the root ops if None

        @return: GRAPH_TUPLE of (nx.DiGraph(), node number of the graph root, operator count)
        """
        root = count
        root_op = root_op if root_op else self.sample_root_helper(max_op, max_depth)

        builder = _GraphBuilder(self, count, max_op, max_depth)
        builder.G.add_node(root, label=root_op)
        select_downstream = ['None'] * self.N_SELECT_SLOTS if select_limit else None
        builder.grow(root_op, cur_depth=0, select_downstream=select_downstream)
        return builder.G, root, builder.op_count

    def generate_task(self) -> Tuple[GRAPH_TUPLE, TASK]:
        """
        generate a random task graph and the corresponding task

        a base subtask is generated first; then, up to self.max_switch times with
        probability self.switch_threshold each, the current task becomes one branch
        of a new conditional (switch) task
        :return: the random task graph tuple and task tuple
        """
        whens = dict()

        # generate a subtask graph and the actual task
        subtask_graph = self.subtask_graph_generator(
            count=0,
            max_op=self.max_op,
            max_depth=self.max_depth,
            select_limit=self.select_limit,
        )
        count = subtask_graph[2] + 1  # start a new subtask graph node number according to old graph
        subtask, whens, attr_whens = tg.subtask_generation(
            env_spec=self.env_spec,
            subtask_graph=subtask_graph,
            get_sig=compute_get_signature(subtask_graph[0]),
            existing_whens=whens,
            is_cond=False,
        )

        for _ in range(self.max_switch):
            if self.rng.random() < self.switch_threshold:  # if add switch
                new_task_graph = self.subtask_graph_generator(
                    count=count,
                    max_op=self.max_op,
                    max_depth=self.max_depth,
                    select_limit=self.select_limit,
                )
                count = new_task_graph[2] + 1

                conditional = self.subtask_graph_generator(
                    count=count,
                    max_op=self.max_op,
                    max_depth=self.max_depth,
                    select_limit=self.select_limit,
                    root_op=self.rng.choice(self.boolean_ops),
                )
                conditional_task, whens, attr_whens = tg.subtask_generation(
                    env_spec=self.env_spec,
                    subtask_graph=conditional,
                    get_sig=compute_get_signature(conditional[0]),
                    existing_whens=whens,
                    existing_attr_whens=attr_whens,
                    is_cond=False,
                )
                new_task, whens, attr_whens = tg.subtask_generation(
                    env_spec=self.env_spec,
                    subtask_graph=new_task_graph,
                    get_sig=compute_get_signature(new_task_graph[0]),
                    existing_whens=whens,
                    existing_attr_whens=attr_whens,
                    is_cond=False,
                )

                if self.rng.random() < self.SWITCH_IF_PROB:  # randomly assign the do_if and do_else branches
                    do_if, do_if_task = subtask_graph, subtask
                    do_else, do_else_task = new_task_graph, new_task
                else:
                    do_if, do_if_task = new_task_graph, new_task
                    do_else, do_else_task = subtask_graph, subtask
                subtask_graph = self.switch_generator(conditional, do_if, do_else)
                count = subtask_graph[2] + 1
                subtask = tg.switch_generation(conditional_task, do_if_task, do_else_task, whens)
        return subtask_graph, subtask
