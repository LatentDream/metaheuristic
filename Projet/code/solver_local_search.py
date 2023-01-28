import numpy as np
import copy
import time
import random

search_time = 30
n_restarts = 50
alpha = 0.99
temperature_init = 1000
max_simulated_iterations = 50


def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    temperature = temperature_init

    start_time = time.time()
    elapsed_time = 0
    solution = generate_random_solution(eternity_puzzle)
    cost_solution = eternity_puzzle.get_total_n_conflict(solution)
    best_solution = solution
    best_cost = cost_solution

    while elapsed_time < search_time:

        solution = generate_random_solution(eternity_puzzle)
        cost_solution = eternity_puzzle.get_total_n_conflict(solution)

        for i in range(max_simulated_iterations):

            neighboorhood = get_neighborhood_two_swap_and_rotations(
                solution, eternity_puzzle
            )
            candidate = random.choice(neighboorhood)
            cost_candidate = eternity_puzzle.get_total_n_conflict(candidate)

            delta = cost_solution - cost_candidate
            probability = max(np.exp(-delta / temperature), 0.01)

            # If the candidate is improving, we accept it as the new solution to explore
            if delta < 0:
                solution = candidate
                cost_solution = cost_candidate

            # If the candidate is not improving, but is accepted by probability, we accept it
            elif np.random.binomial(1, probability):
                solution = candidate
                cost_solution = cost_candidate

            # Is the solution better than best_solution ?
            if cost_solution < best_cost:
                best_solution = solution
                best_cost = cost_solution

            if cost_solution == 0:
                best_solution = solution
                return best_solution, 0

            temperature = temperature * alpha
            elapsed_time = time.time() - start_time

    return best_solution, best_cost


def local_search(eternity_puzzle, validity_function):
    solution = generate_random_solution(eternity_puzzle)
    neighboorhood = get_neighborhood_two_swap_and_rotations(solution, eternity_puzzle)
    valid_neighboorhood = validity_function(eternity_puzzle, neighboorhood)
    while len(valid_neighboorhood) > 0:
        solution = min(
            valid_neighboorhood, key=lambda x: eternity_puzzle.get_total_n_conflict(x)
        )

        neighboorhood = get_neighborhood_two_swap_and_rotations(
            solution, eternity_puzzle
        )
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


def get_neighborhood_two_swap_and_rotations(solution, eternity_puzzle):
    neighbourhood = []
    for i in range(len(solution)):

        rotated = eternity_puzzle.generate_rotation(solution[i])
        for rotated in eternity_puzzle.generate_rotation(solution[i]):
            neighbourhood.append(rotated)

        for j in range(len(solution)):
            if i != j:
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                neighbourhood.append(neighbor)

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


# import numpy as np
# import copy
# import time


# def solve_local_search(eternity_puzzle):
#     """
#     Local search solution of the problem
#     :param eternity_puzzle: object describing the input
#     :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
#         cost is the cost of the solution
#     """
#     search_time = 30
#     alpha = 0.9


#     start_time = time.time()
#     elapsed_time = 0
#     best_solution = generate_random_solution(eternity_puzzle)
#     best_cost = eternity_puzzle.get_total_n_conflict(best_solution)

#     while elapsed_time < search_time:

#         solution = local_search(eternity_puzzle)
#         cost = eternity_puzzle.get_total_n_conflict(solution)


#         if cost < best_cost:
#             best_solution = solution
#         if cost == 0:
#             return best_solution, 0

#         elapsed_time = time.time() - start_time

#     return best_solution, eternity_puzzle.get_total_n_conflict(best_solution)


# def local_search(eternity_puzzle):
#     solution = generate_random_solution(eternity_puzzle)
#     neighboorhood = get_neighborhood_two_swap(solution)
#     valid_neighboorhood = validity_function(eternity_puzzle, neighboorhood)
#     while len(valid_neighboorhood) > 0:
#         solution = min(
#             valid_neighboorhood, key=lambda x: eternity_puzzle.get_total_n_conflict(x)
#         )

#         neighboorhood = get_neighborhood_two_swap(solution)
#         valid_neighboorhood = validity_function(eternity_puzzle, neighboorhood)
#     return solution


# def generate_random_solution(eternity_puzzle):

#     solution = []
#     remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

#     for i in range(eternity_puzzle.n_piece):
#         range_remaining = np.arange(len(remaining_piece))
#         piece_idx = np.random.choice(range_remaining)

#         piece = remaining_piece[piece_idx]

#         permutation_idx = np.random.choice(np.arange(4))

#         piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]

#         solution.append(piece_permuted)

#         remaining_piece.remove(piece)

#     return solution


# def get_neighborhood_two_swap(solution):
#     neighbourhood = []
#     for i in range(len(solution)):
#         for j in range(len(solution)):
#             if i != j:
#                 neighbor = solution.copy()
#                 neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#                 neighbourhood.append(neighbor)

#     return neighbourhood


# def validity_function(eternity_puzzle, neighboorhood):

#     return [
#         n
#         for n in neighboorhood
#         if eternity_puzzle.get_total_n_conflict(n)
#         < eternity_puzzle.get_total_n_conflict(n)
#     ]
