""" procedural task generation.

based on the specified connectivity between operators,
construct random task graphs by adding operators layer by layer
also constructs tasks based on the generated graphs
"""

import random
from typing import Tuple, Union, List

import numpy as np
import networkx as nx

from wisdom.core import Operator, Attribute, Task
from wisdom.utils.auto_task_gen import TaskGenerator
import wisdom.envs.shapenet.task_generator as tg
from wisdom.envs.shapenet.registration import SNEnvSpec

# Tuples of the graph object, the root root_op number, and the number of operators,
# needed to compose graphs in switch generation
GRAPH_TUPLE = Tuple[nx.DiGraph, int, int]
TASK = Tuple[Union[Operator, Attribute], Task]


class SNTaskGenerator(TaskGenerator):
    """
    class for generating random task graphs and tasks based on the specified operator connectivity
    """

    def __init__(self, env_spec: SNEnvSpec):
        """
        constructor for the SNTaskGenerator
        @param env_spec: the task generation configuration,
        see wisdom/envs/shapenet/registration.py for default env_spec
        """
        super().__init__(env_spec)
        self.boolean_ops = self.config['boolean_ops']
        self.max_op = self.config['max_op']
        self.max_depth = self.config['max_depth']
        self.select_limit = self.config['select_limit']
        self.max_switch = self.config['max_switch']
        self.switch_threshold = self.config['switch_threshold']
        self.compare_const_prob = self.config['compare_const_prob']
        self.const_parent_ops = self.config['const_parent_ops']
        self.indexable_get_ops = self.config['indexable_get_ops']

    @staticmethod
    def count_depth_and_op(op: Operator) -> Tuple[int, int]:
        """
        count the number of operators and the depth of the operator graph
        @param op: the root opeartor with nested sucessors
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

    def sample_root_helper(self, max_op, max_depth) -> Operator:
        """
        sample a root operator given constraints on the task complexity
        @param max_op: maximum number of operators allowed
        @param max_depth: maximum depth of the task graph
        @return: the root operator that can complete a subtask graph
        """
        op_depth_limit = self.op_depth_limit
        op_operators_limit = self.op_operators_limit

        depth_filter = [op for op, v in op_depth_limit.items() if (v <= max_depth) and (op in self.root_ops)]
        both_filter = [op for op in depth_filter if op_operators_limit[op] <= max_op]
        if not both_filter:
            depth = {op: v for op, v, in op_depth_limit.items() if op in self.root_ops}
            root_op = {op: v for op, v, in op_operators_limit.items() if op in self.root_ops}
            raise ValueError(f'The specified task complexity is too low given the available root operators\n'
                             f'The minimum depth of the specified root operators is: {depth}\n'
                             f'The minimum depth of the specified root operators is: {root_op}')
        return np.random.choice(both_filter)

    def sample_children_helper(self, op_name, op_count, max_op, cur_depth, max_depth):
        """
        helper function to ensure the task complexity is bounded, and return the sampled child operator
        max_depth is a tighter bound than max_op since operators have to complete their subtask graph
        :param op_name: the current operator
        :param op_count: the current number of operators
        :param max_op: the maximum number of operators allowed
        :param cur_depth: the current depth
        :param max_depth: the maximum depth of the task graph
        :return: a randomly sampled operator to follow the parent node
        """
        op_depth_limit = self.op_depth_limit
        op_operators_limit = self.op_operators_limit
        downstream_ops = self.op_dict[op_name]["downstream"]

        min_add_depth_filter = {op: (max_depth - (cur_depth + self.op_depth_limit[op] - 1)) for op in downstream_ops if
                                (cur_depth + op_depth_limit[op] <= max_depth)}
        min_add_op_filter = {op: (max_op - (op_count + op_operators_limit[op])) for op in downstream_ops if
                             (op_count + op_operators_limit[op] <= max_op)}

        complexity_downstream = [op_depth_limit[op] for op in downstream_ops]
        min_complexity_downstream = [op for op in downstream_ops if op_depth_limit[op] == min(complexity_downstream)]
        if "sample_dist" in self.op_dict[op_name]:  # sample dist overrides complexity bounding
            return np.random.choice(self.op_dict[op_name]["downstream"], p=self.op_dict[op_name]["sample_dist"])
        if not min_add_depth_filter:
            return np.random.choice(min_complexity_downstream)

        if max(min_add_depth_filter.values()) > 0:
            # if added operator sub-graph can fit and have left over depth
            filtered_ops = [op for op, diff in min_add_depth_filter.items() if diff > 0]
            both_filter = {k: v for k, v in min_add_op_filter.items() if k in filtered_ops}
            if both_filter:
                if max(both_filter.values()) > 0:
                    both_filter = [k for k, v in both_filter.items() if v > 0]
                    return np.random.choice(both_filter)
                elif max(both_filter.values()) == 0:
                    both_filter = [k for k, v in both_filter.items() if v == 0]
                    return np.random.choice(both_filter)
            return np.random.choice(min_complexity_downstream)
        elif max(min_add_depth_filter.values()) == 0:
            # if only certain operator sub-graphs can fit
            filtered_ops = [op for op, diff in min_add_depth_filter.items() if diff == 0]
            both_filter = {k: v for k, v in min_add_op_filter.items() if k in filtered_ops}
            if both_filter:
                if max(both_filter.values()) >= 0:
                    both_filter = [k for k, v in both_filter.items() if v == max(both_filter.values())]
                    return np.random.choice(both_filter)
            # cannot do anything about bounding, just complete the minimum task graph
            return np.random.choice(min_complexity_downstream)
        else:
            # cannot do anything about bounding, just complete the minimum task graph
            return np.random.choice(min_complexity_downstream)

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

        n_downstream = self.op_dict[op_name]["n_downstream"]  # how many children need to be sampled
        if n_downstream == 1:
            return [self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)]
        elif op_name == 'Select':
            children = list()  # append children operators

            if cur_depth + 2 > max_depth or op_count + 2 > max_op:
                return ['None' for _ in range(n_downstream)]
            else:
                op_sampled = False
                if select_op:  # if Select is followed by a Get operator
                    if select_downstream is None:
                        select_downstream = self.op_dict['Select']['downstream']
                    for _ in range(n_downstream):
                        if np.random.random() < 0.8:  # make sure not too many operators follow select
                            children.append('None')
                        else:
                            if select_downstream and not op_sampled:  # if the list is not empty, sample a downstream op
                                get = random.choice(select_downstream)
                                children.append(get)
                                op_sampled = True
                                select_downstream.remove(get)
                            else:
                                children.append('None')
                    return children
                else:
                    return ['None' for _ in range(n_downstream)]
        else:
            if self.op_dict[op_name]["same_children_op"]:
                child = self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth)
                return [child for _ in range(n_downstream)]
            else:
                return [self.sample_children_helper(op_name, op_count, max_op, cur_depth, max_depth) for _ in
                        range(n_downstream)]

    def branch_generator(
            self,
            G: nx.DiGraph,
            root_op: str,
            op_count: int,
            max_op: int,
            cur_depth: int,
            max_depth: int,
            select_op=False,
            select_downstream=None,
    ) -> int:
        """
        modifies G in place
        function to complete a branch of subtask graph based on a root operator,
        @param G:
        @param root_op:
        @param op_count:
        @param max_op:
        @param cur_depth:
        @param max_depth:
        @param select_op:
        @param select_downstream:
        @return: the current operator count
        """
        # function to complete a branch of subtask graph based on a root operator,
        # modifies G in place, and returns the leaf node number
        if root_op == 'None':
            return None  # don't add any nodes or edges
        elif root_op == 'CONST':
            return op_count  # don't add any nodes or edges
        else:
            # Exist operator always follows a select with operator as child
            select_op = True if root_op == 'Exist' else select_op

            children = self.sample_children_op(
                op_name=root_op,
                op_count=op_count + 1,
                max_op=max_op,
                cur_depth=cur_depth + 1,
                max_depth=max_depth,
                select_op=select_op,
                select_downstream=select_downstream
            )
            cur_depth += 1  # increment cur_depth count

            if root_op == 'Select' and any('Get' in c for c in children):
                # stop generating overly complex tasks where selects with operator children follow more operators
                select_downstream = ['None'] * 4
                select_op = False

            parent = op_count
            compare_const_prob, const_parent_ops = self.compare_const_prob, self.const_parent_ops
            if root_op in const_parent_ops:
                # current environment only allows comparing category and location const
                # do not compare constants if comparing view_angle and object
                # remove this check if there's mapping for object/view_angle idx
                if not any(op not in self.indexable_get_ops for op in children):
                    if np.random.random() < compare_const_prob:
                        # if the random number is less than the probability
                        # then change one of the indexable Get operators to CONST
                        idx = random.choice(range(len(children)))
                        children[idx] = 'CONST'
                else:
                    if compare_const_prob == 1.0:
                        # force compare to CONST by changing children to indexable Get operators
                        if self.op_dict[root_op]['same_children_op']:
                            n_downstream = self.op_dict[root_op]["n_downstream"]
                            child = random.choice(self.indexable_get_ops)
                            children = [child for _ in range(n_downstream)]
                        else:
                            children = [
                                random.choice(self.indexable_get_ops)
                                if node not in self.indexable_get_ops else node
                                for node in children
                            ]
                        idx = random.choice(range(len(children)))
                        children[idx] = 'CONST'
            for op in children:  # loop over sampled children and modify the graph in place
                if op != 'None':  # if the operator is an operator
                    child = op_count + 1
                    # recursively generate branches based on the child operator
                    # op_count is incremented based on how many nodes were added in the child branch call
                    op_count = self.branch_generator(
                        G, op, child,
                        max_op, cur_depth, max_depth,
                        select_op, select_downstream
                    )
                    G.add_node(child, label=op)  # modify the graph
                    G.add_edge(parent, child)
            return op_count

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
        @param count: the root_op number of the root
        @param max_op: the maximum number of operators allowed in the task graph
        @param max_depth: the maximum depth of the task graph
        @param select_limit: whether to add operator after selects. if True, then constants are sampled.
        If false, then operators could be sampled
        @param root_op: the root operator

        @return: GRAPH_TUPLE of (nx.DiGraph(), node number of the graph root, operator count)
        """
        # initialize the graph and save the root_op number of the root
        G = nx.DiGraph()
        root = count

        op_count = count

        root_op = root_op if root_op else self.sample_root_helper(max_op, max_depth)
        G.add_node(op_count, label=root_op)

        select_downstream = ['None'] * 4 if select_limit else None

        op_count = self.branch_generator(
            G, root_op, op_count,
            max_op, 0, max_depth,
            select_downstream=select_downstream
        )
        return G, root, op_count

    def generate_task(self) -> Tuple[GRAPH_TUPLE, TASK]:
        """
        function to generate a random task graph and corresponding task
        :param max_switch: the maximum number of switch operators allowed
        :param switch_threshold: float in [0,1], how likely a switch operator is sampled, higher is more likely
        :param max_op: max number of operators allowed
        :param max_depth: max depth of the task graph
        :param select_limit: whether to add operator after selects. if True, then constants are sampled.
        If false, then operators could be sampled
        :return: The random task graph tuple and task tuple
        """
        count = 0
        whens = dict()

        # generate a subtask graph and the actual task
        subtask_graph = self.subtask_graph_generator(
            count=count,
            max_op=self.max_op,
            max_depth=self.max_depth,
            select_limit=self.select_limit,

        )
        subtask, whens = tg.subtask_generation(self.env_spec, subtask_graph, existing_whens=whens)
        count = subtask_graph[2] + 1  # start a new subtask graph node number according to old graph
        for _ in range(self.max_switch):
            if random.random() < self.switch_threshold:  # if add switch
                new_task_graph = self.subtask_graph_generator(
                    count=count,
                    max_op=self.max_op,
                    max_depth=self.max_depth,
                    select_limit=self.select_limit
                )
                count = new_task_graph[2] + 1
                conditional = self.subtask_graph_generator(
                    count=count,
                    max_op=self.max_op,
                    max_depth=self.max_depth,
                    select_limit=self.select_limit,
                    root_op=random.choice(self.boolean_ops)
                )
                conditional_task, whens = tg.subtask_generation(self.env_spec, conditional, existing_whens=whens)
                if random.random() < 0.5:  # randomly split the do_if and do_else tasks
                    do_if = subtask_graph
                    do_if_task = subtask
                    do_else = new_task_graph
                    do_else_task, whens = tg.subtask_generation(self.env_spec, do_else, existing_whens=whens)
                else:
                    do_if = new_task_graph
                    do_if_task, whens = tg.subtask_generation(self.env_spec, do_if, existing_whens=whens)
                    do_else = subtask_graph
                    do_else_task = subtask
                subtask_graph = self.switch_generator(conditional, do_if, do_else)
                count = subtask_graph[2] + 1
                subtask = tg.switch_generation(conditional_task, do_if_task, do_else_task, whens)
        return subtask_graph, subtask
