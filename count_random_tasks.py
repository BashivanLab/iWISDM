import numpy as np
import matplotlib.pyplot as plt


def factorial(num):
    if num == 0 or num == 1:
        return 1
    else:
        return num * factorial(num - 1)


def combinations(n, k):
    if k < 0 or k > n:
        return 0
    else:
        return factorial(n) // (factorial(k) * factorial(n - k))


def sum_combinations(n, k):
    result = 0
    for i in range(1, k + 1):
        result += combinations(n, i)
    return result


def count_tree_structures(depth, ng, nc, nb):
    # return the total number of tree structures and select operators
    # for now only consider with 0/1 select operator
    if depth == 3:
        return ng * nc * nb, 2
    elif depth == 4:
        return ng * ng * nc * nb, 2
    elif depth == 7:
        return (ng * nc * nb) ** 3, 6
    elif depth == 8:
        return (ng * nc * nb) ** 3 * (ng + 1) * ng, 6


def count_total_tasks(n_trees, ns, n_frames):
    return n_trees * sum_combinations(n_frames, ns)


if __name__ == "__main__":
    n_getattr_operators = 2
    n_comparison_operators = 2  # IsSame, NotSame
    n_boolean_operators = 2  # And, Or

    min_frames = 1
    max_frames = 6
    total_tasks = np.zeros((max_frames - min_frames, 4))
    total_trees = np.zeros((max_frames - min_frames, 4))

    plt.figure()
    for n_frames in range(min_frames, max_frames):
        for i, depth in enumerate([3, 4, 7, 8]):
            n_trees, ns = count_tree_structures(depth, n_getattr_operators, n_comparison_operators, n_boolean_operators)
            total_trees[n_frames - 1, i] = n_trees
            total_tasks[n_frames - 1, i] = count_total_tasks(n_trees, ns, n_frames)
        plt.plot([3, 4, 7, 8], total_tasks[n_frames - 1], label="n_frames %d" % n_frames)
    plt.legend()
    plt.ylabel("total tasks")
    plt.xlabel("depth")
    plt.savefig("/mnt/store1/xiaoxuan/sanity_check/count_random_task.png")

    plt.figure()
    plt.plot([3, 4, 7, 8], total_trees[0, :])
    plt.ylabel("total trees")
    plt.xlabel("depth")
    plt.savefig("/mnt/store1/xiaoxuan/sanity_check/count_random_trees.png")
