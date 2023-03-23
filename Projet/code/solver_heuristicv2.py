import numpy as np
import math
import copy
import itertools


GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    solution_grid = [
        [(0, 0, 0, 0) for _ in range(eternity_puzzle.board_size)]
        for _ in range(eternity_puzzle.board_size)
    ]

    edges_and_corners = []

    for piece in eternity_puzzle.piece_list:
        if GRAY in piece:
            remaining_piece.remove(piece)
            edges_and_corners.append(piece)

    # Search for the best layout for edges and corners only
    for i in range(eternity_puzzle.board_size):
        for j in range(eternity_puzzle.board_size):

            if (
                i == 0
                or j == 0
                or i == eternity_puzzle.board_size - 1
                or j == eternity_puzzle.board_size - 1
            ):
                print(edges_and_corners)

                best_cost = math.inf
                for piece in edges_and_corners:
                    for rotated_piece in eternity_puzzle.generate_rotation(piece):
                        tested_solution = solution_grid.copy()
                        tested_solution[i][j] = rotated_piece
                        cost = get_current_edges_cost(tested_solution, eternity_puzzle)

                        if cost <= best_cost:
                            best_cost = cost
                            best_piece = piece
                            best_oriented_piece = rotated_piece

                solution_grid[i][j] = best_oriented_piece
                edges_and_corners.remove(best_piece)

    # Search for the best layout with the remaining inner pieces:
    for i in range(eternity_puzzle.board_size):
        for j in range(eternity_puzzle.board_size):
            if (
                i != 0
                and j != 0
                and i != eternity_puzzle.board_size - 1
                and j != eternity_puzzle.board_size - 1
            ):

                best_cost = math.inf

                for piece in remaining_piece:

                    for rotated_piece in eternity_puzzle.generate_rotation(piece):

                        tested_solution = solution_grid.copy()
                        tested_solution[i][j] = rotated_piece
                        cost = get_inner_cost(tested_solution, eternity_puzzle)

                        if cost <= best_cost:
                            best_cost = cost
                            best_piece = piece
                            best_oriented_piece = rotated_piece

                if len(remaining_piece) > 0:
                    solution_grid[i][j] = best_oriented_piece
                    remaining_piece.remove(best_piece)

    # Flatten the grid to a list
    solution = []
    for row in solution_grid:
        for piece in row:
            solution.append(piece)

    print(eternity_puzzle.piece_list)
    print(solution)

    return (solution, eternity_puzzle.get_total_n_conflict(solution))


def get_current_edges_cost(solution, eternity_puzzle):

    n_conflict = 0
    for i in range(eternity_puzzle.board_size):

        if solution[0][i][SOUTH] != GRAY:
            n_conflict += 1

        if solution[eternity_puzzle.board_size - 1][i][NORTH] != GRAY:
            n_conflict += 1

        if solution[i][eternity_puzzle.board_size - 1][EAST] != GRAY:
            n_conflict += 1

        if solution[i][0][WEST] != GRAY:
            n_conflict += 1

        if i > 0 and solution[i][0][SOUTH] != solution[i - 1][0][NORTH]:
            n_conflict += 1

        if (
            i > 0
            and solution[i][eternity_puzzle.board_size - 1][SOUTH]
            != solution[i - 1][eternity_puzzle.board_size - 1][NORTH]
        ):
            n_conflict += 1

        if i > 0 and solution[0][i][WEST] != solution[i - 1][0][EAST]:
            n_conflict += 1

        if (
            i > 0
            and solution[eternity_puzzle.board_size - 1][i][WEST]
            != solution[eternity_puzzle.board_size - 1][i - 1][EAST]
        ):
            n_conflict += 1

    return n_conflict


def get_inner_cost(solution, eternity_puzzle):

    n_conflict = 0

    n_placed = len(solution)
    n_lines = (
        n_placed // eternity_puzzle.board_size + 1
        if n_placed % eternity_puzzle.board_size != 0
        else n_placed // eternity_puzzle.board_size
    )
    n_columns = (
        eternity_puzzle.board_size
        if n_lines > 1
        else n_placed % eternity_puzzle.board_size
    )

    for j in range(n_lines):
        for i in range(n_columns):
            if eternity_puzzle.board_size * j + i < n_placed:
                k = eternity_puzzle.board_size * j + i
                k_west = eternity_puzzle.board_size * j + (i - 1)
                k_south = eternity_puzzle.board_size * (j - 1) + i
                # print(
                #     "\n solution {} \n n_placed {} \n n_lines {}\n n_columns {}\n i : {} \n j : {} \n k : {} \n k_west {} \n k_south {}".format(
                #         solution, n_placed, n_lines, n_columns, i, j, k, k_west, k_south
                #     )
                # )

                if i > 0 and solution[k][WEST] != solution[k_west][EAST]:
                    n_conflict += 1

                if j > 0 and solution[k][SOUTH] != solution[k_south][NORTH]:
                    n_conflict += 1

    return n_conflict
