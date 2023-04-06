"""
Guillaume Blanché
Guillaume Thibault

The proposed work is inspired from : 
https://www.researchgate.net/publication/267412224_Solving_Eternity-II_puzzles_with_a_tabu_search_algorithm
"""

import numpy as np
from copy import deepcopy
import time
import random
from utils.utils import *
from math import inf

########################### PARAMETERS ####################################
tabu_length = 360  # Length of tabu list
I1 = 400  # Maximum number of iterations of Phase I
I2 = 400  # Number of non-improving iterations for simulated annealing
I3 = 3500  # Number of non-improving iterations for perturbation
beta1 = 0.5  # Ratio of pairs of homogeneous pieces in tabu sear
beta2 = 6000  # Maximum number of pairs of homogeneous pieces in tabu search
alpha = 0.99  # Scaling value in the cooling schedule
T_0 = 100  # Initial temperature in simulated annealing
T_E = 0  # Final temperature
gamma = 0.5  # Ratio of pairs of homogeneous pieces in perturbation
############################################################################

# Time limits
start_time = time.time()
search_time_border = 10
search_time = 60 * 2 - search_time_border


def solve_advanced(e):
    """
    Your solver for the problem
    :param e: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    # Phase I : Border Construction
    best_border, _ = TabuSearch_border(e)
    print(best_border)
    remaining_pieces = Remaining_Pieces(e, best_border)

    visualize(e, best_border, "Border_after_tabu")

    # Phase II : Full Construction
    best_solution, best_solution_cost = TabuSearch_inner(
        e, best_border, remaining_pieces
    )

    return best_solution, best_solution_cost


# Tabu Search for the border of the board
def TabuSearch_border(e: EternityPuzzle):

    # Randomly initialize the border
    current_solution = generate_random_solution(e)
    best_solution = current_solution
    best_cost = getBorderCost(e, best_solution)
    tabu_list = []
    tabu_list.append(best_solution)

    for _ in range(I1):
        # Initialize the best candidate for this iteration
        best_border_candidate = None
        best_border_candidate_cost = inf

        # Generate the neighborhood
        neighborhood = getNeighbors_border(e, current_solution)

        # Evaluate each candidate solution in the neighborhood
        for candidate in neighborhood:

            # Check if the candidate solution is tabu
            if is_tabu_type1(tabu_list, candidate) or is_tabu_type2(
                tabu_list, candidate
            ):
                continue

            # Evaluate the candidate solution
            candidate_cost = getBorderCost(e, candidate)

            # Update the best candidate if necessary
            if candidate_cost < best_border_candidate_cost:
                best_border_candidate = candidate
                best_border_candidate_cost = candidate_cost

        # Update the current solution and best solution if necessary
        if best_border_candidate_cost < best_cost:
            current_solution = best_border_candidate
            best_solution = best_border_candidate
            best_cost = best_border_candidate_cost

        if len(tabu_list) > tabu_length:
            tabu_list.pop(0)
        tabu_list.append(current_solution)

    return best_solution, best_cost


# Tabu Search for the inner pieces of the board
def TabuSearch_inner(e, best_border, remaining_piece):

    best_solution = generate_random_innner_solution(e, best_border, remaining_piece)
    best_solution_cost = e.get_total_n_conflict(best_solution)
    best_candidate = best_solution

    tabuList = [best_solution]

    elapsed_time = start_time - time.time()
    while elapsed_time < search_time:

        neighborhood = getNeighbors_inner(best_candidate, e)

        best_candidate = neighborhood[0]
        best_candidate_cost = e.get_total_n_conflict(best_candidate)

        for candidate in neighborhood:
            a = e.get_total_n_conflict(candidate)
            if (not candidate in tabuList) and (a < best_candidate_cost):
                best_candidate = candidate
                best_candidate_cost = a

        if best_candidate_cost < best_solution_cost:
            best_solution = best_candidate
            best_solution_cost = best_candidate_cost

        tabuList.append(best_candidate)

        if len(tabuList) > tabu_length:
            tabuList = tabuList[1:]

        elapsed_time = time.time() - start_time

        if best_solution_cost == 0:
            return best_solution, best_solution_cost

    return best_solution, best_solution_cost


# Function to generate a random solution
def generate_random_solution(e):

    solution = []
    remaining_piece = deepcopy(e.piece_list)

    for i in range(e.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = e.generate_rotation(piece)[permutation_idx]
        solution.append(piece_permuted)
        remaining_piece.remove(piece)

    return solution


# Function to generate a random solution starting from a border already completed
def generate_random_innner_solution(e, best_border, remaining_piece):

    solution = list_to_grid(best_border, e.board_size, e.board_size)
    x = 1
    y = 1

    for i in range(len(remaining_piece)):

        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = e.generate_rotation(piece)[permutation_idx]
        print(x)
        print(y)
        solution[x][y] = piece_permuted
        remaining_piece.remove(piece)

        if y < e.board_size - 1:
            y += 1
        else:
            y = 1
            x += 1

    solution = grid_to_list(solution)

    return solution


# 2 swap with rotations neighbourhood
def getNeighbors_inner(solution, e):

    solution = list_to_grid(solution, e.board_size, e.board_size)
    neighbourhood = []

    for i in range(1, e.board_size - 1):
        for j in range(1, e.board_size - 1):

            neighbor1 = solution.copy()
            neighbor2 = grid_to_list(solution.copy())

            # Rotated inner pieces
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

            # 2 swap with rotations between inner pieces
            if i != j:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                for rotated_piece1 in e.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in e.generate_rotation(neighbor2[j]):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)

    return neighbourhood


def getNeighbors_border(e, solution):
    # Il faut juste placer les bords, pas les pièces à l'intérieur

    solution = list_to_grid(solution, e.board_size, e.board_size)
    neighbourhood = []

    for i in range(e.board_size):
        for j in range(e.board_size):

            neighbor1 = solution.copy()
            neighbor2 = grid_to_list(solution.copy())

            # Rotated inner pieces
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

            # 2 swap with rotations between inner pieces
            if i != j:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                for rotated_piece1 in e.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in e.generate_rotation(neighbor2[j]):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)

    return neighbourhood


def getBorderCost(e, border):

    border_copy = deepcopy(border)
    border_copy = list_to_grid(border_copy, e.board_size, e.board_size)
    # Set all the inner pieces to be black to ignore them in the cost
    for i in range(1, e.board_size - 1):
        for j in range(1, e.board_size - 1):
            border_copy[i][j] = (23, 23, 23, 23)

    border_copy = grid_to_list(border_copy)
    border_cost = e.get_total_n_conflict(border_copy)

    return border_cost


def is_tabu_type1(tabu_list, candidate_solution):
    # TODO
    return False


def is_tabu_type2(tabu_list, candidate_solution):
    # TODO
    return False


def Remaining_Pieces(e, border):

    remaining_piece = []
    border_grid = list_to_grid(border, e.board_size, e.board_size)

    for i in range(1, e.board_size - 1):
        for j in range(1, e.board_size - 1):
            remaining_piece.append(border_grid[i][j])

    return remaining_piece
