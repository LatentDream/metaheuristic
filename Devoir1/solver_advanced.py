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


t_init = 1


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
    return local_search_with_restart(pcstp)


# Initialization functions #
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

def first_improving_neighbor(pcstp, solution):
    """
    Neighboorhood function:
    """
    selected_edges = []
    for e in solution:
        a, b, w = e
        selected_edges.append((a, b, w))
        if len(selected_edges) == 100:
            break

    not_selected_edges = []

    candidate_edges = []
    for edge in pcstp.network.edges(data=True):
        if len(candidate_edges) == 100:
            break
        if random.choice([True, False]):
            candidate_edges.append(edge)

    for e in candidate_edges:
        if len(not_selected_edges) == 100:
            break
        a, b, w = e
        if (
            (a, b, w) not in selected_edges
            and (b, a, w) not in selected_edges
            and (b, a, w) not in not_selected_edges
        ):
            not_selected_edges.append((a, b, w))

    for edge in selected_edges:
        neighbour1 = deepcopy(selected_edges)
        neighbour1.remove(edge)
        if pcstp.get_solution_cost(neighbour1) < pcstp.get_solution_cost(
            solution
        ) and pcstp.verify_solution(neighbour1):
            return neighbour1

    for edge in not_selected_edges:
        neighbour2 = deepcopy(solution)
        neighbour2.append(edge)
        if pcstp.get_solution_cost(neighbour2) < pcstp.get_solution_cost(
            solution
        ) and pcstp.verify_solution(neighbour2):
            return neighbour2

    return neighbour1

def local_search_with_restart(pcstp, max_time=19 * 60):

    start_time = time.time()
    elapsed_time = 0

    solution = generate_random_solution(pcstp)

    valid_neighbor = first_improving_neighbor(pcstp, solution)
    temperature = t_init
    best_solution = solution

    while elapsed_time < max_time:

        candidate = valid_neighbor
        delta = pcstp.get_solution_cost(candidate) - pcstp.get_solution_cost(solution)
        probability = max(
            np.exp(-delta / temperature), 0.01
        )  # the probability of the degradation is always greater than 1%

        if delta < 0:
            solution = candidate
        elif np.random.binomial(1, probability):
            solution = candidate

        if pcstp.get_solution_cost(solution) < pcstp.get_solution_cost(best_solution):
            best_solution = solution
            print("best_solution found")

        temperature = t_init * (max_time - elapsed_time) / (max_time)

        valid_neighbor = first_improving_neighbor(pcstp, solution)

        elapsed_time = time.time() - start_time

    best_solution = format_solution(best_solution)
    return best_solution


def format_solution(solution):
    S = []
    for a, b, _ in solution:
        S.append((a, b))
    return S
