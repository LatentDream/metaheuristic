import numpy as np
import math
import copy
from utils.utils import *
import itertools
from copy import deepcopy

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def solve_heuristic(e: EternityPuzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    solution = [(BLACK, BLACK, BLACK, BLACK) for _ in range(e.n_piece)]
    corners = [piece for piece in e.piece_list if piece_type(piece) == "corner"]
    edges = [piece for piece in e.piece_list if piece_type(piece) == "edge"]
    inner = [piece for piece in e.piece_list if piece_type(piece) == "inner"]
    color_count = {color: 0 for color in e.build_color_dict()}

    edge_idx = [
        i
        for i in range(1, e.n_piece - 1)
        if (
            i < e.board_size - 1
            or i % e.board_size == 0
            or e.n_piece - i < e.board_size
            or (i + 1) % e.board_size == 0
        )
        and (i != e.n_piece - e.board_size and i != e.board_size - 1)
    ]

    inner_idx = [
        i
        for i in range(e.n_piece)
        if not (
            i <= e.board_size - 1
            or i % e.board_size == 0
            or e.n_piece - i <= e.board_size
            or (i + 1) % e.board_size == 0
        )
    ]

    (
        solution[0],
        solution[e.board_size - 1],
        solution[e.n_piece - e.board_size],
        solution[e.n_piece - 1],
    ) = (corners[0], corners[1], corners[2], corners[3])

    # Generate the 4 distinct corner configurations
    corner_geometries = generate_corner_geometries(e, solution)

    best_cost_geometry = math.inf
    best_solution = None

    # Build the best solution with edges and inner pieces for each geometry
    for geometry in corner_geometries:
        solution = deepcopy(geometry)

        corners = [piece for piece in e.piece_list if piece_type(piece) == "corner"]
        edges = [piece for piece in e.piece_list if piece_type(piece) == "edge"]
        inner = [piece for piece in e.piece_list if piece_type(piece) == "inner"]
        color_count = {color: 0 for color in e.build_color_dict()}

        # set edges
        for position in edge_idx:
            best_cost = math.inf

            for piece in edges:
                for rotated_piece in e.generate_rotation(piece):
                    tested_solution = solution.copy()
                    tested_solution[position] = rotated_piece
                    cost = e.get_total_n_conflict(tested_solution)
                    if cost <= best_cost:
                        best_cost = cost
                        best_piece = piece
                        best_oriented_piece = rotated_piece
                        solution[position] = best_oriented_piece

            if len(edges) > 0:
                edges.remove(best_piece)

        # set inner pieces
        for position in inner_idx:
            best_cost = math.inf
            for piece in inner:
                for rotated_piece in e.generate_rotation(piece):
                    tested_solution = solution.copy()
                    tested_solution[position] = rotated_piece
                    cost = e.get_total_n_conflict(tested_solution)

                    if cost < best_cost:
                        best_cost = cost
                        best_piece = piece
                        best_oriented_piece = rotated_piece
                        solution[position] = best_oriented_piece
                        color_count = update_color_counter(
                            color_count, solution[position]
                        )

                    elif cost == best_cost and evaluate_color_counter(
                        update_color_counter(color_count, piece)
                    ) < evaluate_color_counter(color_count):
                        best_cost = cost
                        best_piece = piece
                        best_oriented_piece = rotated_piece
                        solution[position] = best_oriented_piece
                        color_count = update_color_counter(
                            color_count, solution[position]
                        )

            if len(inner) > 0:
                inner.remove(best_piece)

        cost_geometry = e.get_total_n_conflict(solution)
        if cost_geometry < best_cost_geometry:
            best_solution = solution
            best_cost_geometry = cost_geometry

    # Return the best solution of the best geometry
    return (best_solution, best_cost_geometry)


def generate_corner_geometries(e: EternityPuzzle, solution):
    c1, c2, c3, c4 = (
        solution[0],
        solution[e.board_size - 1],
        solution[e.n_piece - e.board_size],
        solution[e.n_piece - 1],
    )

    solution = [(BLACK, BLACK, BLACK, BLACK) for _ in range(e.n_piece)]

    g1 = deepcopy(solution)
    g2 = deepcopy(solution)
    g3 = deepcopy(solution)
    g4 = deepcopy(solution)

    g1[0], g1[e.board_size - 1], g1[e.n_piece - e.board_size], g1[e.n_piece - 1] = (
        c1,
        c2,
        c3,
        c4,
    )
    g2[0], g2[e.board_size - 1], g2[e.n_piece - e.board_size], g2[e.n_piece - 1] = (
        c1,
        c3,
        c2,
        c4,
    )
    g3[0], g3[e.board_size - 1], g3[e.n_piece - e.board_size], g3[e.n_piece - 1] = (
        c1,
        c2,
        c4,
        c3,
    )

    g4[0], g4[e.board_size - 1], g4[e.n_piece - e.board_size], g4[e.n_piece - 1] = (
        c1,
        c4,
        c3,
        c2,
    )

    return (
        orient_corners(e, g1),
        orient_corners(e, g2),
        orient_corners(e, g3),
        orient_corners(e, g4),
    )


def update_color_counter(color_count, piece):
    color_count_copy = deepcopy(color_count)
    for color in piece:
        color_count_copy[color] += 1
    return color_count_copy


def evaluate_color_counter(color_count):
    return sum(color_count**2 for color_count in color_count.values())
