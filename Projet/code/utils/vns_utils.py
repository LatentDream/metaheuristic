import matplotlib.pyplot as plt
from typing import Dict, List
from utils.utils import *
from tqdm import tqdm
import solver_heuristic_layer
import numpy as np
import copy
import time



def prioritise_neighborhhod(e: EternityPuzzle, i, j, sigma, inverted, debug=False):
     #? Generate Gaussian filter 
    sigma=0.5
    muu=0

    i, j = 2*i, 2*j

    kernel_size = e.board_size * 2
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    # lower normal part of gaussian
    normal = 1/(2.0 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gaussian = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    if inverted:
        gaussian = (gaussian - 1) * -1

    #? filter on the i, j area
    half_board_range = e.board_size // 2

    i_lower = max(0, i-half_board_range)
    i_upper = i_lower + e.board_size
    if i_upper >= kernel_size:
        i_lower -= i_upper - kernel_size
        i_upper -= i_upper - kernel_size

    j_lower = max(0, j-half_board_range)
    j_upper = j_lower + e.board_size
    if j_upper >= kernel_size:
        j_lower -= j_upper - kernel_size
        j_upper -= j_upper - kernel_size

    neighborhood = gaussian[i_lower:i_upper, j_lower:j_upper]

    print(neighborhood)

    if debug:
        plt.imshow(neighborhood, cmap='binary')
        plt.colorbar()
        plt.savefig("selected_neighborhood")
        plt.close()

    return neighborhood


def swap(e, solution, k, debug=False):
    return gloton_place_removed_pieces(*choose(e, solution, k, nb_piece_to_choose=2, debug=debug))


def rotate(e, solution, k, debug=False):
    return gloton_place_removed_pieces(*choose(e, solution, k, nb_piece_to_choose=1, debug=debug))


def choose(e: EternityPuzzle, solution, probabilities, nb_piece_to_choose: int=2, debug: bool=False):

    solution = deepcopy(solution) # To make sure we are not affecting the input solution

    # Remove k pieces with the most conflict
    conflict_position, idx_to_nb_conflict = get_conflict_positions(e, solution, return_nb_conflict=True)

    # Sample the distribution
    removed_pieces_idx = []
    while len(removed_pieces_idx) != nb_piece_to_choose:
        idx_list = [i for i in range(0, e.board_size * e.board_size)]
        idx = np.random.choice(idx_list, p=probabilities)
        if piece_type(solution[idx]) != INNER:
            continue
        removed_pieces_idx.append(idx)
        del idx_to_nb_conflict[idx]

    # Remove the selected pieces
    removed_pieces = []
    for idx in removed_pieces_idx:
        removed_pieces.append(solution[idx])
        solution[idx] = (BLACK, BLACK, BLACK, BLACK) # Set hole to black; If their is a problem, we'll be able to see it.

    # Return all the paramter for the rebuild function
    return e, solution, removed_pieces, removed_pieces_idx, debug

