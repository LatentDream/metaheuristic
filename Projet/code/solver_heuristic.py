import numpy as np
import math
import copy


def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    solution = []

    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        best_cost = math.inf
        
        for piece in remaining_piece:
            for rotated_piece in eternity_puzzle.generate_rotation(piece):
                tested_solution = solution.copy()
                tested_solution.append(rotated_piece)
                cost = get_current_cost(tested_solution, eternity_puzzle)

                if cost <= best_cost:
                    best_cost = cost
                    best_piece = piece
                    best_oriented_piece = rotated_piece

        solution.append(best_oriented_piece)
        remaining_piece.remove(best_piece)

    return (solution, eternity_puzzle.get_total_n_conflict(solution))


GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def get_current_cost(solution, eternity_puzzle):

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
                print(
                    "\n solution {} \n n_placed {} \n n_lines {}\n n_columns {}\n i : {} \n j : {} \n k : {} \n k_west {} \n k_south {}".format(
                        solution, n_placed, n_lines, n_columns, i, j, k, k_west, k_south
                    )
                )

                if i == 0 and solution[k][WEST] != GRAY:
                    n_conflict += 1

                if i == eternity_puzzle.board_size - 1 and solution[k][EAST] != GRAY:
                    n_conflict += 1

                if i > 0 and solution[k][WEST] != solution[k_west][EAST]:
                    n_conflict += 1

                if i == eternity_puzzle.board_size - 1 and solution[k][EAST] != GRAY:
                    n_conflict += 1

                if j > 0 and solution[k][SOUTH] != solution[k_south][NORTH]:
                    n_conflict += 1

                if j == 0 and solution[k][SOUTH] != GRAY:
                    n_conflict += 1

                if j == eternity_puzzle.board_size - 1 and solution[k][NORTH] != GRAY:
                    n_conflict += 1

    return n_conflict
