"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""

from typing import List, Tuple
from network import PCSTP
import networkx as nx
import numpy as np
import random
import time
from copy import deepcopy


t_init = 1000


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
    return ls_simulated_annealing(pcstp)


# Initialization functions
def generate_random_solution(pcstp):
    """
    Generate a random solution : select randomly if each edge is accepted or not
    """
    return [
        edge for edge in pcstp.network.edges(data=True) if random.choice([True, False])
    ]


def generate_random_valid_solution(pcstp):
    """
    Generate a random solution : select randomly if each edge is accepted or not
    """
    solution = []
    for edge in pcstp.network.edges(data=True):
        candidate = deepcopy(solution)
        candidate.append(edge)
        if random.choice([True, False]) and pcstp.verify_solution(candidate):
            solution.append(edge)
    return solution


def ls_simulated_annealing(pcstp, max_time=19 * 60):

    start_time = time.time()
    elapsed_time = 0
    temperature = t_init

    solution = generate_random_valid_solution(pcstp)
    best_solution = solution

    neighboorhood = neighbour_function(pcstp, solution)
    valid_neigboorhood = validity_function(pcstp, neighboorhood, solution)

    while elapsed_time < max_time:

        candidate = random.choices(valid_neigboorhood)[0]

        delta = pcstp.get_solution_cost(candidate) - pcstp.get_solution_cost(solution)
        # the probability of the degradation is always greater than 1% :
        probability = max(np.exp(-delta / temperature), 0.01)

        if delta < 0:
            solution = candidate

        elif np.random.binomial(1, probability):
            solution = candidate

        if pcstp.get_solution_cost(solution) < pcstp.get_solution_cost(best_solution):
            best_solution = solution
            print(
                "best_solution found : cost {}".format(
                    pcstp.get_solution_cost(best_solution)
                )
            )

        temperature = t_init * (max_time - elapsed_time) / (max_time)

        neighboorhood = neighbour_function(pcstp, solution)
        valid_neigboorhood = validity_function(pcstp, neighboorhood, solution)

        elapsed_time = time.time() - start_time

    best_solution = format_solution(best_solution)
    return best_solution


def format_solution(solution):
    S = []
    for a, b, _ in solution:
        S.append((a, b))
    return S


def neighbour_function(pcstp, solution):

    N = []
    selected_edges = solution
    random.shuffle(solution)

    # Create the list of the edges that are not in the solution, without repetition
    not_selected_edges = []
    for e in pcstp.network.edges(data=True):
        a, b, w = e
        if (
            (a, b, w) not in selected_edges
            and (b, a, w) not in selected_edges
            and (b, a, w) not in not_selected_edges
            and random.random() > 0.9
        ):
            not_selected_edges.append((a, b, w))

    for edge in selected_edges:
        neighbour1 = deepcopy(selected_edges)
        neighbour1.remove(edge)
        if (
            pcstp.get_solution_cost(neighbour1) < pcstp.get_solution_cost(solution)
            or random.random() > 0.95
        ):
            N.append(neighbour1)

    for edge in not_selected_edges:
        neighbour2 = deepcopy(solution)
        neighbour2.append(edge)
        if (
            pcstp.get_solution_cost(neighbour2) < pcstp.get_solution_cost(solution)
            or random.random() > 0.95
        ):
            N.append(neighbour2)

    random.shuffle(N)
    return N


def validity_function(pcstp, neighborhood, solution):
    V = []
    for neighbor in neighborhood:
        if pcstp.verify_solution(neighbor):
            return [neighbor]

    if len(V) == 0:
        V.append(generate_random_valid_solution(pcstp))
    return V
