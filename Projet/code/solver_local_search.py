import numpy as np
import time
import random
import solver_heuristic
from copy import deepcopy
from utils.utils import *


def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    solution = generate_random_solution(eternity_puzzle)
    return local_search(eternity_puzzle, solution, search_time=20 * 60)


def local_search(e, solution, search_time=30):
    """Simulated annealing local search"""

    start_time = time.time()

    alpha = 0.999
    temperature_init = 10
    temperature = temperature_init
    cost_solution = e.get_total_n_conflict(solution)
    best_solution, best_cost = solution, cost_solution

    while True:
        neighboorhood = get_neighborhood(e, solution)
        candidate = random.choices(neighboorhood)[0]
        cost_candidate = e.get_total_n_conflict(candidate)
        delta = e.get_total_n_conflict(solution) - cost_candidate
        probability = min(1, max(np.exp(-delta / temperature), 0.01))

        if delta < 0:
            solution = candidate
            cost_solution = cost_candidate

        elif np.random.binomial(1, probability):
            solution = candidate
            cost_solution = cost_candidate

        if cost_solution < best_cost:
            best_solution = solution
            best_cost = cost_solution
            print(best_cost)

        temperature = temperature * alpha
        if time.time() - start_time > search_time:
            break

    return best_solution, e.get_total_n_conflict(best_solution)


def generate_random_solution(e: EternityPuzzle):

    solution = []
    remaining_piece = deepcopy(e.piece_list)

    for _ in range(e.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = e.generate_rotation(piece)[permutation_idx]
        solution.append(piece_permuted)
        remaining_piece.remove(piece)

    return solution


def get_neighborhood(e, solution):

    solution_list = solution.copy()
    solution_grid = list_to_grid(
        solution.copy(),
        e.board_size,
        e.board_size,
    )
    neighbourhood = []

    for i in range(e.board_size):
        for j in range(e.board_size):

            neighbor1 = solution_grid.copy()
            neighbor2 = solution_list.copy()

            # Rotated pieces
            for rotated_piece in e.generate_rotation(neighbor1[i][j]):
                if rotated_piece != neighbor1[i][j]:

                    neighbor1[i][j] = rotated_piece
                    neighbor1 = grid_to_list(neighbor1)
                    neighbourhood.append(neighbor1)
                    neighbor1 = list_to_grid(
                        neighbor1,
                        e.board_size,
                        e.board_size,
                    )

            # 2 swap with rotations
            if i != j:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                for rotated_piece1 in e.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in e.generate_rotation(neighbor2[j]):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)
    return neighbourhood


def is_improving_validity_function(eternity_puzzle, neighboorhood):
    "Accept all the improving neighbors"
    return [
        n
        for n in neighboorhood
        if eternity_puzzle.get_total_n_conflict(n)
        < eternity_puzzle.get_total_n_conflict(n)
    ]


def accept_all_validity_function(neighboorhood):
    """
    All the neighboors are valid (used in simulated annealing)
    """
    return neighboorhood
