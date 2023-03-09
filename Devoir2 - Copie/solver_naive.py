import time
from typing import List
from tsptw import TSPTW
import networkx as nx
import itertools


def solve(tsptw: TSPTW) -> List[int]:
    """Naive solution to the problem: connect 2 terminal nodes via shortest path.

    Args:
        tsptw (TSPTW): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    nodes = [i for i in range(1, tsptw.num_nodes)]
    start = time.time()
    best_solution = None
    best_solution_cost = float("inf")
    for permutation in itertools.permutations(nodes):
        if time.time() - start >= 60:
            break
        candidate_solution = [0] + list(permutation) + [0]
        candidate_solution_feasible = tsptw.verify_solution(candidate_solution)
        if not candidate_solution_feasible:
            continue
        candidate_solution_cost = tsptw.get_solution_cost(candidate_solution)
        if candidate_solution_cost < best_solution_cost:
            best_solution_cost = candidate_solution_cost
            best_solution = candidate_solution

    print(best_solution)
    return best_solution
