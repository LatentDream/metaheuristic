import numpy as np
import time
import random
from copy import deepcopy
from utils.utils import *
import solver_heuristic_layer

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def solve_local_search(e: EternityPuzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    random.seed(1234)
    # solution = generate_random_solution(e)
    solution = solver_heuristic_layer.solve_heuristic(e)[0]
    return local_search(e, solution, search_time=20 * 60)


def local_search(
    e: EternityPuzzle, solution, search_time=30, temperature_init=10, alpha=0.99
):
    """Simulated annealing local search"""

    start_time = time.time()

    temperature = temperature_init
    cost_solution = e.get_total_n_conflict(solution)
    best_solution, best_cost = solution, cost_solution
    print(f"Local Search cost: {best_cost}", end="\r")

    min_proba_counter = 0
    while True:
        neighboorhood = get_neighborhood(e, solution)
        improving_candidates = [
            c for c in neighboorhood if e.get_total_n_conflict(c) < cost_solution
        ]

        if len(improving_candidates) > 0:
            candidate = random.choice(improving_candidates)
        else:
            candidate = random.choices(neighboorhood)[0]

        cost_candidate = e.get_total_n_conflict(candidate)
        delta = cost_solution - cost_candidate

        probability = max(np.exp(-delta / temperature), 0.01)

        if delta < 0:
            solution = candidate
            cost_solution = cost_candidate

        elif np.random.binomial(1, probability):
            solution = candidate
            cost_solution = cost_candidate

        if cost_solution < best_cost:
            best_solution = solution
            best_cost = cost_solution
            print(f"Local Search cost: {best_cost}", end="\r")

        temperature = temperature * alpha

        if probability == 0.01:
            min_proba_counter += 1

        # If probability is fixed at the minimum : restart on a random_solution
        if min_proba_counter > 100:
            solution = generate_random_solution(e)
            cost_solution = e.get_total_n_conflict(solution)
            temperature = temperature_init
            min_proba_counter = 0

        if (time.time() - start_time) >= search_time:
            break

    return best_solution, best_cost


def get_neighborhood(e: EternityPuzzle, solution):
    neighbourhood = [solution]

    conflict_positions = get_conflict_positions(e, solution)
    random.shuffle(conflict_positions)

    for i in conflict_positions:
        neighbor1 = deepcopy(solution)

        # rotate pieces with conflict
        for rotated_piece in e.generate_rotation(neighbor1[i])[1:]:
            neighbor1[i] = rotated_piece
            neighbourhood.append(neighbor1)

        # 2-swap a piece with a conflict with another one
        neighbor2 = deepcopy(solution)
        for j in conflict_positions:
            if j != i:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                neighbourhood.append(neighbor2)

        # if len(neighbourhood) >= 2000:
        #     break

    return neighbourhood
