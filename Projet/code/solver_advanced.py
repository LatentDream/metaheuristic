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
from utils import *

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
search_time_border = 60
search_time = 60 * 2 - search_time_border


def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    # Phase I : The border
    best_border, remaining_piece = TabuSearch_border(eternity_puzzle)
    print(remaining_piece)

    # Phase II : The inner pieces
    best_solution, best_solution_cost = TabuSearch_inner(
        eternity_puzzle, best_border, remaining_piece
    )

    return best_solution, best_solution_cost


# Tabu Search for the edges of the board
def TabuSearch_border(eternity_puzzle):

    # Randomly initialize the border
    best_border = generate_random_border(eternity_puzzle)
    best_border_cost = getBorderCost(best_border, eternity_puzzle)
    best_border_candidate = best_border
    # visualize(eternity_puzzle, best_border)

    # Simple Tabu Search for the border
    tabuList = [best_border]

    elapsed_time = start_time - time.time()
    while elapsed_time < search_time_border:

        neighborhood = getNeighbors_border(best_border_candidate, eternity_puzzle)

        best_border_candidate = neighborhood[0]
        best_border_candidate_cost = getBorderCost(
            best_border_candidate, eternity_puzzle
        )

        for border_candidate in neighborhood:
            a = getBorderCost(border_candidate, eternity_puzzle)
            if (not border_candidate in tabuList) and (a < best_border_candidate_cost):
                best_border_candidate = border_candidate
                best_border_candidate_cost = a

        if best_border_candidate_cost < best_border_cost:
            best_border = best_border_candidate
            best_border_cost = best_border_candidate_cost

        tabuList.append(best_border_candidate)

        if len(tabuList) > tabu_length:
            tabuList = tabuList[1:]

        if best_border_cost == 0:
            return best_border, best_border_cost

        elapsed_time = time.time() - start_time

    # List of all the pieces that are not already used for the border
    remaining_piece = []
    for piece in eternity_puzzle.piece_list:
        flag = False
        for rotated_piece in eternity_puzzle.generate_rotation(piece):
            if rotated_piece in best_border:
                flag = True
                break
        if not flag:
            remaining_piece.append(piece)

    visualize(eternity_puzzle, best_border)

    return best_border, remaining_piece


# Tabu Search for the inner pieces of the board
def TabuSearch_inner(eternity_puzzle, best_border, remaining_piece):

    best_solution = generate_random_innner_solution(
        eternity_puzzle, best_border, remaining_piece
    )
    best_solution_cost = eternity_puzzle.get_total_n_conflict(best_solution)
    best_candidate = best_solution

    tabuList = [best_solution]

    elapsed_time = start_time - time.time()
    while elapsed_time < search_time:

        neighborhood = getNeighbors_inner(best_candidate, eternity_puzzle)

        best_candidate = neighborhood[0]
        best_candidate_cost = eternity_puzzle.get_total_n_conflict(best_candidate)

        for candidate in neighborhood:
            a = eternity_puzzle.get_total_n_conflict(candidate)
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
def generate_random_border(eternity_puzzle):

    solution = []
    remaining_piece = deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]
        solution.append(piece_permuted)
        remaining_piece.remove(piece)

    return solution


# Function to generate a random solution starting from a border already completed
def generate_random_innner_solution(eternity_puzzle, best_border, remaining_piece):

    solution = list_to_grid(
        best_border, eternity_puzzle.board_size, eternity_puzzle.board_size
    )
    x = 1
    y = 1

    for i in range(len(remaining_piece)):

        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]
        print(x)
        print(y)
        solution[x][y] = piece_permuted
        remaining_piece.remove(piece)

        if y < eternity_puzzle.board_size - 1:
            y += 1
        else:
            y = 1
            x += 1

    solution = grid_to_list(solution)

    return solution


# 2 swap with rotations neighbourhood
def getNeighbors_inner(solution, eternity_puzzle):

    solution = list_to_grid(
        solution, eternity_puzzle.board_size, eternity_puzzle.board_size
    )
    neighbourhood = []

    for i in range(1, eternity_puzzle.board_size - 1):
        for j in range(1, eternity_puzzle.board_size - 1):

            neighbor1 = solution.copy()
            neighbor2 = grid_to_list(solution.copy())

            # Rotated inner pieces
            for rotated_piece in eternity_puzzle.generate_rotation(neighbor1[i][j]):
                if rotated_piece != neighbor1[i][j]:

                    neighbor1[i][j] = rotated_piece
                    neighbor1 = grid_to_list(neighbor1)
                    neighbourhood.append(neighbor1)
                    neighbor1 = list_to_grid(
                        neighbor1,
                        eternity_puzzle.board_size,
                        eternity_puzzle.board_size,
                    )

            # 2 swap with rotations between inner pieces
            if i != j:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                for rotated_piece1 in eternity_puzzle.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in eternity_puzzle.generate_rotation(
                        neighbor2[j]
                    ):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)

    return neighbourhood


def getNeighbors_border(solution, eternity_puzzle):
    # Il faut juste placer les bords, pas les pièces à l'intérieur

    solution = list_to_grid(
        solution, eternity_puzzle.board_size, eternity_puzzle.board_size
    )
    neighbourhood = []

    for i in range(eternity_puzzle.board_size):
        for j in range(eternity_puzzle.board_size):

            neighbor1 = solution.copy()
            neighbor2 = grid_to_list(solution.copy())

            # Rotated inner pieces
            for rotated_piece in eternity_puzzle.generate_rotation(neighbor1[i][j]):
                if rotated_piece != neighbor1[i][j]:

                    neighbor1[i][j] = rotated_piece
                    neighbor1 = grid_to_list(neighbor1)
                    neighbourhood.append(neighbor1)
                    neighbor1 = list_to_grid(
                        neighbor1,
                        eternity_puzzle.board_size,
                        eternity_puzzle.board_size,
                    )

            # 2 swap with rotations between inner pieces
            if i != j:
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                for rotated_piece1 in eternity_puzzle.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in eternity_puzzle.generate_rotation(
                        neighbor2[j]
                    ):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)

    return neighbourhood


def getBorderCost(border, eternity_puzzle):

    border_copy = deepcopy(border)
    border_copy = list_to_grid(
        border_copy, eternity_puzzle.board_size, eternity_puzzle.board_size
    )

    for i in range(1, eternity_puzzle.board_size - 1):
        for j in range(1, eternity_puzzle.board_size - 1):
            border_copy[i][j] = (23, 23, 23, 23)

    border_copy = grid_to_list(border_copy)
    border_cost = eternity_puzzle.get_total_n_conflict(border_copy)

    return border_cost
