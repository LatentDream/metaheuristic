"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 
"""

from typing import List, Tuple
from network import PCSTP
import networkx as nx
import numpy as np
import math
import random
import time
from copy import deepcopy


def solve(pcstp: PCSTP) -> List[Tuple[int]]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph
            are excluded
    """
    return local_search_with_restart(pcstp, 100)


# Initialization functions #
def generate_random_solution(pcstp):
    """
    Generate a random solution : select randomly if each edge is accepted or not
    """

    solution = []
    for edge in pcstp.network.edges(data=True):
        random_bool = random.choice([True, False])
        if random_bool:
            solution.append(edge)
    return solution


# Neighboorhood function #
def generate_neighboorhood(pcstp, solution):
    """
    Neighboorhood function


        Verify that there is no cycles
    """
    not_selected_edges = [
        e for e in pcstp.network.edges(data=True) if e not in solution
    ]
    neighbourhood = []

    for edge in solution:

        neighbour1 = deepcopy(solution)
        neighbour1.remove(edge)
        if pcstp.verify_solution(neighbour1):
            neighbourhood.append(neighbour1)

    for edge in not_selected_edges:
        neighbour2 = deepcopy(solution)
        neighbour2.append(edge)
        if pcstp.verify_solution(neighbour2):
            neighbourhood.append(neighbour2)

    return neighbourhood


# Validity functions #
def is_improving_validity_function(pcstp, neighboorhood, solution):
    """
    Return only the list of neighbours that are improving the evaluation cost
    """
    return [
        n
        for n in neighboorhood
        if pcstp.get_solution_cost(n) <= pcstp.get_solution_cost(solution)
    ]


def accept_all_neighboors(neighboorhood):
    """
    All the neighboors are valid (used in simulated annealing)
    """
    return neighboorhood


def local_search(pcstp):

    start_time = time.time()

    solution = generate_random_solution(pcstp)
    neighboorhood = generate_neighboorhood(pcstp, solution)
    valid_neighboorhood = is_improving_validity_function(pcstp, neighboorhood, solution)

    while len(valid_neighboorhood) > 0:

        solution = min(valid_neighboorhood, key=lambda x: pcstp.get_solution_cost(x))

        neighboorhood = generate_neighboorhood(pcstp, solution)
        valid_neighboorhood = is_improving_validity_function(
            pcstp, neighboorhood, solution
        )

    return solution


def local_search_with_restart(pcstp, n_restart):

    start_time = time.time()
    best_solution = generate_random_solution(pcstp)

    for i in range(n_restart):

        # Note that the seed for the generation of the random instance changes at each iteration
        solution = local_search(pcstp)

        if pcstp.get_solution_cost(solution) < pcstp.get_solution_cost(best_solution):
            best_solution = solution

        print("RESTART: ", i)

    best_solution = format_solution(best_solution)
    print(best_solution)
    return best_solution


def format_solution(solution):
    S = []
    for a, b, w in solution:
        S.append((a, b))
    return S
