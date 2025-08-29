import networkx as nx
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Optional


def min_n_for_k_pairs(K: int) -> int:
    """Return smallest n such that C(n,2) >= K."""
    if K <= 0:
        return 0
    n = 2
    while (n * (n - 1)) // 2 < K:
        n += 1
    return n


def compute_node_signatures(
        G: nx.DiGraph,
        nodes_subset=None,
        node_type_key: str = "label"
):
    """
    Compute bottom-up signatures for nodes in G (or in a subgraph defined by nodes_subset).
    Signature for node n: (label, tuple(sorted(child_signatures)))
    Only node labels (node_type_key) and structure are considered.
    Returns: dict node -> signature (hashable tuple)
    """
    # operate on the requested subgraph
    subG = G.subgraph(list(nodes_subset)).copy() if nodes_subset is not None else G

    def node_label(n):
        return subG.nodes[n].get(node_type_key, str(n))

    # produce a processing order: children first
    if nx.is_directed_acyclic_graph(subG):
        topo = list(nx.topological_sort(subG))
        order = list(reversed(topo))  # leaves first
    else:
        raise ValueError('task graph should be DAGs')

    signatures = dict()
    for n in order:
        # gather child signatures (if any). If a child hasn't been processed treat it
        # as a leaf with signature (label, ()).
        child_sigs = []
        for c in subG.successors(n):
            if c in signatures:
                child_sigs.append(signatures[c])
            else:
                child_sigs.append((node_label(c), ()))
        # sort to make signature order-independent across siblings
        child_sigs_sorted = tuple(sorted(child_sigs))
        sig = (node_label(n), child_sigs_sorted)
        signatures[n] = sig
    return signatures, order


def group_equivalent_subgraphs(
        G: nx.DiGraph,
        nodes_subset: Optional[Iterable[Any]] = None,
        node_type_key: str = "label"
) -> Dict[Tuple, List[Any]]:
    """
    Group nodes (roots of subgraphs) by their bottom-up signature (label + structure).
    Returns a dict: signature -> list of nodes in G that share that signature.
    """
    sigs, order = compute_node_signatures(G, nodes_subset=nodes_subset, node_type_key=node_type_key)
    groups: Dict[Tuple, List[Any]] = defaultdict(list)
    for node, sig in sigs.items():
        groups[sig].append(node)
    return dict(groups), order


def find_equivalent_pairs_across_graphs(
        G1: nx.DiGraph,
        G2: nx.DiGraph,
        nodes_subset1: Optional[Iterable[Any]] = None,
        nodes_subset2: Optional[Iterable[Any]] = None,
        node_type_key: str = "label"
) -> List[Tuple[Any, Any]]:
    """
    Return list of (node_from_G1, node_from_G2) pairs where the rooted subgraphs
    (label+structure) are equivalent according to compute_node_signatures.
    """
    sigs1 = compute_node_signatures(G1, nodes_subset=nodes_subset1, node_type_key=node_type_key)
    sigs2 = compute_node_signatures(G2, nodes_subset=nodes_subset2, node_type_key=node_type_key)

    # invert sigs2 for fast lookup
    inv2: Dict[Tuple, List[Any]] = defaultdict(list)
    for n2, s2 in sigs2.items():
        inv2[s2].append(n2)

    pairs: List[Tuple[Any, Any]] = []
    for n1, s1 in sigs1.items():
        matches = inv2.get(s1)
        if matches:
            for n2 in matches:
                pairs.append((n1, n2))

    return pairs


def equivalent_groups_across_graphs(
        G1: nx.DiGraph,
        G2: nx.DiGraph,
        nodes_subset1: Optional[Iterable[Any]] = None,
        nodes_subset2: Optional[Iterable[Any]] = None,
        node_type_key: str = "label"
) -> Dict[Tuple, Tuple[List[Any], List[Any]]]:
    """
    Return a dict mapping signature -> (list_of_nodes_in_G1, list_of_nodes_in_G2)
    for signatures that appear in either graph (empty list if none in one graph).
    """
    sigs1 = compute_node_signatures(G1, nodes_subset=nodes_subset1, node_type_key=node_type_key)
    sigs2 = compute_node_signatures(G2, nodes_subset=nodes_subset2, node_type_key=node_type_key)

    groups: Dict[Tuple, Tuple[List[Any], List[Any]]] = {}
    all_sigs = set(sigs1.values()) | set(sigs2.values())
    inv1: Dict[Tuple, List[Any]] = defaultdict(list)
    inv2: Dict[Tuple, List[Any]] = defaultdict(list)

    for n1, s1 in sigs1.items():
        inv1[s1].append(n1)
    for n2, s2 in sigs2.items():
        inv2[s2].append(n2)

    for sig in all_sigs:
        groups[sig] = (inv1.get(sig, []), inv2.get(sig, []))

    return groups
