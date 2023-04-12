from typing import Dict, List
import numpy as np
import math
import copy
from solver_genetic import genetic_algorithm_border
from utils.utils import *
import itertools
from copy import deepcopy
import time
from tqdm import tqdm

BLACK = 23

def solve_lns(e: EternityPuzzle):
    """
    Large neighborhood search
    """

    #! Debug option
    debug = True

    random.seed(1998)

    # Solve the border
    border_time = 1 * 60
    pop_size = 20
    mutation_rate = 0
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 10
    elite_size = 1

    initial_solution, nb_conflict = genetic_algorithm_border(
        e,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        elite_size=elite_size,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=border_time,
        debug_visualization=debug
    )

    print("Border final cost : {}".format(nb_conflict))
    
    # initial_solution = generate_random_solution(e)
    visualize(e, initial_solution, "debogging_border")

    return lns(e, initial_solution, search_time=1*60, debug=debug)



def lns(e: EternityPuzzle, solution, search_time=30, debug=False):

    visualize(e, solution, "debug/before_destruction")
    start_time = time.time()
    tic = start_time
    best_solution = deepcopy(solution)
    n_conflit_best_solution = e.get_total_n_conflict(best_solution)

    print(f"Solving inner puzzle ...")
    with tqdm(total=search_time) as progress_bar:

        while True:

            new_solution = rebuild(*destroy(e, solution, neighborhood_size=5, allow_adjacent=False, debug_visualization=debug))

            if (nb_confllict_new_sol := e.get_total_n_conflict(new_solution)) < e.get_total_n_conflict(solution):
                solution = new_solution
                if nb_confllict_new_sol < n_conflit_best_solution:
                    best_solution = deepcopy(new_solution)
                    n_conflit_best_solution = nb_confllict_new_sol

            if (tac := time.time()) - start_time < search_time:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                break

    return best_solution, n_conflit_best_solution



def destroy(e: EternityPuzzle, solution, neighborhood_size=4, allow_adjacent=False, debug_visualization=False):

    solution = deepcopy(solution) # To make sure we are not affecting the input solution

    # Remove k pieces with the most conflict
    conflict_position, idx_to_nb_conflict = get_conflict_positions(e, solution, return_nb_conflict=True)
    
    # Find the probability of removing a piece base on the number of conflicts
    def get_proba(idx_to_nb_conflict: Dict):
        pieces_with_most_conflict = [(idx, idx_to_nb_conflict[idx] + 0.5) for idx in idx_to_nb_conflict.keys() if piece_type(solution[idx]) == INNER]
        idx_conflic = [idx for idx, _ in pieces_with_most_conflict]
        probabilities = np.array([nb_conflict for _, nb_conflict in pieces_with_most_conflict])
        probabilities = probabilities / np.sum(probabilities)
        return idx_conflic, probabilities

    # Sample the distribution
    removed_pieces_idx = []
    while len(removed_pieces_idx) != neighborhood_size:
        idx_conflic, probabilities = get_proba(idx_to_nb_conflict)
        if len(idx_conflic) == 0: break
        idx = np.random.choice(idx_conflic, p=probabilities)
        # Check if it allow to remove the piece
        if not allow_adjacent and is_adjacent(e, idx, removed_pieces_idx): 
            del idx_to_nb_conflict[idx]
            continue
        removed_pieces_idx.append(idx)
        del idx_to_nb_conflict[idx]

    # Remove the selected pieces
    removed_pieces = []
    for idx in removed_pieces_idx:
        removed_pieces.append(solution[idx])
        solution[idx] = (BLACK, BLACK, BLACK, BLACK) # Set hole to black; If their is a problem, we'll be able to see it.

    # Return all the paramter for the rebuild function
    return e, solution, removed_pieces, removed_pieces_idx, debug_visualization



def rebuild(e: EternityPuzzle, destroyed_solution, removed_pieces, removed_pieces_idx, debug_visualization=False):
    # Realidxated removed pieces optimally to the holes
    if debug_visualization:
        visualize(e, destroyed_solution, "debug/rebuilding")

    for _ in range(len(removed_pieces_idx)):

        k = random.choice([i for i in range(len(removed_pieces_idx))])
        idx_to_fill = removed_pieces_idx[k]
        del removed_pieces_idx[k]

        # Find piece that have the best fit
        idx_of_piece_use_to_fill_hole, piece_to_fill_hole, min_number_of_conflit = -1, None, 5
        for i, piece in enumerate(removed_pieces):
            best_fit, number_of_conflit = find_best_fit(e, destroyed_solution, idx_to_fill, piece)
            if number_of_conflit < min_number_of_conflit:
                idx_of_piece_use_to_fill_hole, piece_to_fill_hole, min_number_of_conflit = i, best_fit, number_of_conflit
        del removed_pieces[idx_of_piece_use_to_fill_hole]

        # Fill the hole
        destroyed_solution[idx_to_fill] = piece_to_fill_hole

    if debug_visualization:
        visualize(e, destroyed_solution, "debug/debug")

    return destroyed_solution



def find_best_fit(e: EternityPuzzle, solution, idx, piece):
    
    piece_best_fit = sorted([(rotated_piece, get_number_conflict(e, solution, idx, rotated_piece)) for rotated_piece in e.generate_rotation(piece)], key=lambda x: x[1])

    return piece_best_fit[0]



def get_number_conflict(e: EternityPuzzle, solution, idx, piece):
    """ Swipe with side not supported """

    i , j = idx % e.board_size, idx // e.board_size
    idx_south =  e.board_size * (j - 1) + i
    idx_north =  e.board_size * (j + 1) + i
    idx_east =   e.board_size * j + (i - 1)
    idx_west =   e.board_size * j + (i + 1)

    nb_conflict = 0

    if piece[WEST]  != solution[idx_west][EAST]:
        nb_conflict += 1
    if piece[NORTH] != solution[idx_north][SOUTH]:
        nb_conflict += 1
    if piece[EAST]  != solution[idx_east][WEST]:
        nb_conflict += 1
    if piece[SOUTH] != solution[idx_south][NORTH]:
        nb_conflict += 1

    return nb_conflict



def is_adjacent(e: EternityPuzzle, idx: int, puzzle_pieces_idx: List[int]):
    i , j = idx % e.board_size, idx // e.board_size
    idx_south =  e.board_size * (j - 1) + i
    idx_north =  e.board_size * (j + 1) + i
    idx_east =   e.board_size * j + (i - 1)
    idx_west =   e.board_size * j + (i + 1)

    for piece_idx in puzzle_pieces_idx:
        if idx_north == piece_idx: return True
        if idx_south == piece_idx: return True
        if idx_east == piece_idx: return True
        if idx_west == piece_idx: return True

    return False
