from typing import List
import heapq
from typing import List, Tuple
from tsptw import TSPTW
import time
import networkx as nx
import itertools
import random
import numpy as np
from math import inf, ceil, log2, exp, log


def get_neighbors(tsptw, state):
    neighbors = []
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            # Swap two elements
            neighbor = state.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def generate_random_solution(tsptw: TSPTW):
    chromosome = list(range(1, tsptw.num_nodes))
    random.shuffle(chromosome)
    chromosome = [0] + chromosome + [0]
    return chromosome


def beam_search(tsptw: TSPTW, beam_width: int, time_limit: float) -> np.ndarray:
    start_time = time.time()

    # Initialize the beam with a random solution
    beam = [generate_random_solution(tsptw)]
    solutions = []
    # Iterate until the time limit is exceeded
    while time.time() - start_time < time_limit:
        # Generate the neighbors of the solutions in the beam
        neighbors = []
        for solution in beam:
            neighbors.extend(get_neighbors(tsptw, solution))

        # Keep only the top-k solutions from the neighbors
        neighbors = sorted(
            neighbors, key=lambda s: tsptw.get_solution_cost(s), reverse=True
        )[:beam_width]

        # Check if any of the neighbors satisfy the hard constraints
        new_solutions = [
            neighbor for neighbor in neighbors if tsptw.verify_solution(neighbor)
        ]

        # If new_solutions is empty, generate a new random solution
        if not new_solutions:
            beam = [generate_random_solution(tsptw)]
            continue

        # Update the beam with the new solutions
        beam = new_solutions
        print("{} valid solutions found".format(len(new_solutions)))
        solutions.extend(new_solutions)

    # Return the best solution found so far
    return min(solutions, key=lambda s: tsptw.get_solution_cost(s))


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    return beam_search(tsptw, beam_width=20, time_limit=10)
