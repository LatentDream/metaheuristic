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
import solver_naive

t_init = 100
alpha = 0.99


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
    return [edge for edge in pcstp.network.edges if random.choice([True, False])]


# Neighboorhood function #
def generate_neighboorhood(pcstp, solution):
    """
    Neighboorhood function:


    """
    not_selected_edges = [
        e for e in pcstp.network.edges(data=True) if e not in solution
    ]
    neighbourhood = []

    for edge in solution:

        neighbour1 = deepcopy(solution)
        neighbour1.remove(edge)
        neighbourhood.append(neighbour1)

    for edge in not_selected_edges:
        neighbour2 = deepcopy(solution)
        neighbour2.append(edge)
        neighbourhood.append(neighbour2)

    return neighbourhood

    # for edge in solution:
    #     a,b = edge
    #     for neighbor in pcstp.network.ad[a]:
    #         if (a,neighbor) in solution :
    #             neighbour1 = deepcopy(solution)
    #             neighbour1.remove((a,neighbor))
    #         elif (neighbor,a) in solution:
    #             neighbour1.remove((neighbor,a))

    # ##? Choose a random node in the solution
    # solution = deepcopy(solution)
    # edge = random.choice(solution)
    # node = random.choice(edge)

    # temperature = 0.75

    # ##? Local search -> Look at the neighboor and switch the connection with P(temperautre)
    # for neighboor in pcstp.network.adj[node]:
    #     if random.random() < temperature:
    #         if (node, neighboor) in solution:
    #                 solution.remove((node, neighboor))
    #         elif (neighboor, node) in solution:
    #                 solution.remove((neighboor, node))
    #         else:
    #             ##? Check if a cicle is created
    #             new_connexion = (node, neighboor)
    #             solution.append(new_connexion)
    #             if pcstp.verify_solution(solution) or random.random() < 0.1:
    #                 continue
    #             else:
    #                 solution.remove(new_connexion)


# Validity functions
def is_improving_validity_function(pcstp, neighboorhood, solution):
    """
    Return only the list of neighbours that are improving the evaluation cost
    """
    return [
        n
        for n in neighboorhood
        if pcstp.get_solution_cost(n) <= pcstp.get_solution_cost(solution)
    ]


def accept_all_neighboors(pcstp, neighboorhood):
    """
    All the neighboors are valid (used in simulated annealing)
    """
    validity_neighborhood = [n for n in neighboorhood if pcstp.verify_solution(n)]
    if len(validity_neighborhood) == 0:
        validity_neighborhood.append(solver_naive.solve(pcstp))
    return validity_neighborhood


def local_search_with_restart(pcstp, max_time=3600):

    start_time = time.time()
    elapsed_time = 0

    solution = generate_random_solution(pcstp)
    neighboorhood = generate_neighboorhood(pcstp, solution)
    valid_neighboorhood = accept_all_neighboors(pcstp, neighboorhood)

    temperature = t_init
    best_solution = solution

    while elapsed_time < max_time:

        candidate = random.choices(valid_neighboorhood)[0]
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

        temperature = alpha * temperature
        neighboorhood = generate_neighboorhood(pcstp, solution)
        valid_neighboorhood = accept_all_neighboors(pcstp, neighboorhood)

        elapsed_time = time.time() - start_time

    best_solution = format_solution(best_solution)
    return best_solution


def format_solution(solution):
    S = []
    for a, b, _ in solution:
        S.append((a, b))
    return S
