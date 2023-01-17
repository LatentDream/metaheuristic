import numpy as np
import copy
import time


def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    search_time = 60 * 5

    start_time = time.time()
    elapsed_time = 0
    best_solution = generate_random_solution(eternity_puzzle)
    best_cost = eternity_puzzle.get_total_n_conflict(best_solution)

    while elapsed_time < search_time:

        solution = local_search(eternity_puzzle)
        cost = eternity_puzzle.get_total_n_conflict(solution)
        if cost < best_cost:
            best_solution = solution
        if cost == 0:
            return best_solution, 0

        elapsed_time = time.time() - start_time

    return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)


def local_search(eternity_puzzle):
    solution = generate_random_solution(eternity_puzzle)
    neighboorhood = get_neighborhood_two_swap(solution)
    valid_neighboorhood = validity_function(eternity_puzzle, neighboorhood)
    while len(valid_neighboorhood) > 0:
        solution = min(
            valid_neighboorhood, key=lambda x: eternity_puzzle.get_total_n_conflict(x)
        )

        neighboorhood = get_neighborhood_two_swap(solution)
        valid_neighboorhood = validity_function(eternity_puzzle, neighboorhood)
    return solution


def generate_random_solution(eternity_puzzle):

    solution = []
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)

        piece = remaining_piece[piece_idx]

        permutation_idx = np.random.choice(np.arange(4))

        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]

        solution.append(piece_permuted)

        remaining_piece.remove(piece)

    return solution


def get_neighborhood_two_swap(solution):
    neighbourhood = []
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i != j:
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbourhood.append(neighbor)

    return neighbourhood


def validity_function(eternity_puzzle, neighboorhood):

    return [
        n
        for n in neighboorhood
        if eternity_puzzle.get_total_n_conflict(n)
        < eternity_puzzle.get_total_n_conflict(n)
    ]
