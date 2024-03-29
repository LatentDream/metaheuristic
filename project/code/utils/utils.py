import copy
from eternity_puzzle import EternityPuzzle
import random
import numpy as np
from copy import deepcopy
from matplotlib import pyplot

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

MINUTE = 60

CORNER = "corner"
EDGE = "edge"
INNER = "inner"


# Save a visualisation
def visualize(e: EternityPuzzle, solution, name="visualisation"):
    e.display_solution(solution, name)
    pyplot.close()  # Free memory if a lot of write visualize is done


def piece_type(piece):
    count_gray = piece.count(GRAY)
    return CORNER if count_gray == 2 else EDGE if count_gray == 1 else INNER


# Return a list of the positions of the pieces in conflict
def get_conflict_positions(e, solution, return_nb_conflict=False):
    positions = []

    n_conflit = {i: 0 for i in range(e.board_size * e.board_size)}

    for j in range(e.board_size):
        for i in range(e.board_size):
            k = e.board_size * j + i
            k_east = e.board_size * j + (i - 1)
            k_south = e.board_size * (j - 1) + i

            if i > 0 and solution[k][WEST] != solution[k_east][EAST]:
                positions.append(k)
                n_conflit[k] += 1
                positions.append(k_east)
                n_conflit[k_east] += 1

            if i == 0 and solution[k][WEST] != GRAY:
                positions.append(k)
                n_conflit[k] += 1

            if i == e.board_size - 1 and solution[k][EAST] != GRAY:
                positions.append(k)
                n_conflit[k] += 1

            if j > 0 and solution[k][SOUTH] != solution[k_south][NORTH]:
                positions.append(k)
                n_conflit[k] += 1
                positions.append(k_south)
                n_conflit[k_south] += 1

            if j == 0 and solution[k][SOUTH] != GRAY:
                positions.append(k)
                n_conflit[k] += 1

            if j == e.board_size - 1 and solution[k][NORTH] != GRAY:
                positions.append(k)
                n_conflit[k] += 1

    positions = list(set(positions))

    if return_nb_conflict:
        return positions, n_conflit

    return positions


# Swap the orientation of 2 edges or 2 corners : the GRAY positions are exchanged
def swap_orientations(e: EternityPuzzle, piece1, piece2):
    if piece_type(piece1) == "corner" and piece_type(piece2) == "corner":
        gray_positions_1 = [i for i in range(4) if piece1[i] == GRAY]
        gray_positions_2 = [i for i in range(4) if piece2[i] == GRAY]

        for rotated_piece1 in e.generate_rotation(piece1):
            if (
                rotated_piece1[gray_positions_2[0]] == GRAY
                and rotated_piece1[gray_positions_2[1]] == GRAY
            ):
                piece1 = rotated_piece1

        for rotated_piece2 in e.generate_rotation(piece2):
            if (
                rotated_piece2[gray_positions_1[0]] == GRAY
                and rotated_piece2[gray_positions_1[1]] == GRAY
            ):
                piece1 = rotated_piece2

    if piece_type(piece1) == "edge" and piece_type(piece2) == "edge":
        gray_position_1 = [i for i in range(4) if piece1[i] == GRAY][0]
        gray_position_2 = [i for i in range(4) if piece2[i] == GRAY][0]

        for rotated_piece1 in e.generate_rotation(piece1):
            if rotated_piece1[gray_position_2] == GRAY:
                piece1 = rotated_piece1

        for rotated_piece2 in e.generate_rotation(piece2):
            if rotated_piece2[gray_position_1] == GRAY:
                piece2 = rotated_piece2

    return piece1, piece2


# Given a solution, set the GRAY colors of the corners to the ouside
def orient_corners(e: EternityPuzzle, solution):
    solution[0] = [
        corner
        for corner in e.generate_rotation(solution[0])
        if corner[SOUTH] == GRAY and corner[WEST] == GRAY
    ][0]
    solution[e.board_size - 1] = [
        corner
        for corner in e.generate_rotation(solution[e.board_size - 1])
        if corner[SOUTH] == GRAY and corner[EAST] == GRAY
    ][0]
    solution[e.n_piece - e.board_size] = [
        corner
        for corner in e.generate_rotation(solution[e.n_piece - e.board_size])
        if corner[NORTH] == GRAY and corner[WEST] == GRAY
    ][0]
    solution[e.n_piece - 1] = [
        corner
        for corner in e.generate_rotation(solution[e.n_piece - 1])
        if corner[NORTH] == GRAY and corner[EAST] == GRAY
    ][0]

    return solution


# Given a solution, set the GRAY color of the edges  to the ouside
def orient_edges(e: EternityPuzzle, solution):
    south_edge_idx = [i for i in range(1, e.board_size)]
    west_edge_idx = [
        i for i in range(1, e.n_piece - e.board_size) if i % e.board_size == 0
    ]
    east_edge_idx = [
        i for i in range(e.board_size + 1, e.n_piece - 1) if (i + 1) % e.board_size == 0
    ]
    north_edge_idx = [i for i in range(e.n_piece - e.board_size + 1, e.n_piece)]

    for i in south_edge_idx:
        solution[i] = [
            edge for edge in e.generate_rotation(solution[i]) if edge[SOUTH] == GRAY
        ][0]

    for i in west_edge_idx:
        solution[i] = [
            edge for edge in e.generate_rotation(solution[i]) if edge[WEST] == GRAY
        ][0]
    for i in east_edge_idx:
        solution[i] = [
            edge for edge in e.generate_rotation(solution[i]) if edge[EAST] == GRAY
        ][0]

    for i in north_edge_idx:
        solution[i] = [
            edge for edge in e.generate_rotation(solution[i]) if edge[NORTH] == GRAY
        ][0]


def generate_random_solution(e: EternityPuzzle):
    """Constraints :
    Corners are on the corners and are well oriented
    Edges are on the edges and are well oriented
    """
    solution = [(BLACK, BLACK, BLACK, BLACK) for _ in range(e.n_piece)]

    remaining_piece = e.piece_list

    corners = [piece for piece in remaining_piece if piece_type(piece) == "corner"]
    random.shuffle(corners)

    edges = [piece for piece in remaining_piece if piece_type(piece) == "edge"]
    random.shuffle(edges)

    inner = [
        piece
        for piece in remaining_piece
        if ((piece not in corners) and (piece not in edges))
    ]
    random.shuffle(inner)

    for i in range(e.n_piece):
        # Set corners and orient them
        if (
            i == 0
            or i == e.board_size - 1
            or i == e.n_piece - e.board_size
            or i == e.n_piece - 1
        ):
            piece = corners.pop()
            solution[i] = piece

        # Bottom edges :
        elif (
            i < e.board_size - 1
            or i % e.board_size == 0
            or e.n_piece - i < e.board_size
            or (i + 1) % e.board_size == 0
        ) and (
            i != 0
            and i != e.board_size - 1
            and i != e.n_piece - e.board_size
            and i != e.n_piece - 1
        ):
            piece = edges.pop()
            solution[i] = piece

        else:
            piece = inner.pop()
            solution[i] = e.generate_rotation(piece)[np.random.choice(np.arange(4))]

    orient_corners(e, solution)
    orient_edges(e, solution)

    return solution


def generate_random_inner_solution(e: EternityPuzzle, border):
    """Generate a random solution but keeps the border intact"""

    solution = deepcopy(border)

    # IDs of the inner pieces
    inner_ids = [
        i
        for i in range(e.n_piece)
        if not (
            i <= e.board_size - 1
            or i % e.board_size == 0
            or e.n_piece - i <= e.board_size
            or (i + 1) % e.board_size == 0
        )
    ]
    inner_pieces = [solution[i] for i in inner_ids]
    random.shuffle(inner_pieces)

    for i, position in enumerate(inner_ids):
        solution[position] = random.choice(e.generate_rotation(inner_pieces[i]))

    return solution


# Function to flatten a grid into a list
def grid_to_list(grid):
    return [piece for row in grid for piece in row]


# Function to create a 2D grid from a list
def list_to_grid(e: EternityPuzzle, liste):
    grid = []
    for i in range(e.board_size):
        grid.append(liste[i * e.board_size : i * e.board_size + e.board_size])
    return grid


# ? """ LNS AND VNS Utils """


def find_best_fit(e: EternityPuzzle, solution, idx, piece):
    piece_best_fit = sorted(
        [
            (rotated_piece, get_number_conflict(e, solution, idx, rotated_piece))
            for rotated_piece in e.generate_rotation(piece)
        ],
        key=lambda x: x[1],
    )
    return piece_best_fit[0]


def get_number_conflict(e: EternityPuzzle, solution, idx, piece):
    """Swipe with side not supported"""
    idx_north, idx_east, idx_south, idx_west = get_adjacent_idx(e, idx)
    nb_conflict = 0
    if piece[WEST] != solution[idx_west][EAST] and solution[idx_west][EAST] != BLACK:
        nb_conflict += 1
    if (
        piece[NORTH] != solution[idx_north][SOUTH]
        and solution[idx_north][SOUTH] != BLACK
    ):
        nb_conflict += 1
    if piece[EAST] != solution[idx_east][WEST] and solution[idx_east][WEST] != BLACK:
        nb_conflict += 1
    if (
        piece[SOUTH] != solution[idx_south][NORTH]
        and solution[idx_south][NORTH] != BLACK
    ):
        nb_conflict += 1
    return nb_conflict


def get_adjacent_idx(e: EternityPuzzle, idx):
    i, j = idx % e.board_size, idx // e.board_size
    idx_south = e.board_size * (j - 1) + i
    idx_north = e.board_size * (j + 1) + i
    idx_east = e.board_size * j + (i - 1)
    idx_west = e.board_size * j + (i + 1)
    return idx_north, idx_east, idx_south, idx_west


def gloton_place_removed_pieces(
    e: EternityPuzzle,
    destroyed_solution,
    removed_pieces,
    holes_idx,
    debug_visualization=False,
):
    """O(len(removed_pieces)!)"""

    assert len(removed_pieces) == len(holes_idx)

    if len(holes_idx) == 0:
        return destroyed_solution

    i = 0
    solution_found = []
    while i < len(holes_idx):
        # Try to fill the hole
        hole_idx = holes_idx[i]
        for j, removed_piece in enumerate(removed_pieces):
            best_fit, number_of_conflit = find_best_fit(
                e, destroyed_solution, hole_idx, removed_piece
            )

            # Copy Args
            rebuild_solution = copy.deepcopy(destroyed_solution)
            rebuild_solution[hole_idx] = best_fit
            rebuild_removed_pieces = copy.deepcopy(removed_pieces)
            del rebuild_removed_pieces[j]
            rebuild_holes_idx = copy.deepcopy(holes_idx)
            del rebuild_holes_idx[i]

            # Fill next hole
            rebuild_solution = gloton_place_removed_pieces(
                e, rebuild_solution, rebuild_removed_pieces, rebuild_holes_idx
            )
            solution_found.append(rebuild_solution)

        i += 1

    sorted_solution = sorted(
        [(solution, e.get_total_n_conflict(solution)) for solution in solution_found],
        key=lambda x: x[1],
    )
    best_solution = sorted_solution[0][0]

    return best_solution
