from typing import Dict, List
import numpy as np
import math
import copy
from solver_genetic import genetic_algorithm_border
import solver_heuristic_layer
from utils.utils import *
import itertools
from copy import deepcopy
import time
from tqdm import tqdm


PRIORITY_WEIGHT = 100


def solve_lns(e: EternityPuzzle):
    """
    Large neighborhood search
    """

    #! Debug option
    debug = False

    #? Option
    tabu_queue_size= 5
    lns_search_time = 2 * MINUTE
    neighborhood_size = 20
    allow_adjacent = True

    #?  Solve the border
    border_time = 1 * 5
    pop_size = 30
    mutation_rate = 0
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 10
    elite_size = 2

    random.seed(1998)
    
    print("  [INFO] Solving border ...")
    # initial_solution, nb_conflict = genetic_algorithm_border(e, num_generations=num_generations, no_progress_generations=no_progress_generations, elite_size=elite_size, tournament_size=tournament_size, tournament_accepted=tournament_accepted, pop_size=pop_size, time_limit=border_time, debug_visualization=debug)
    initial_solution, nb_conflict = solver_heuristic_layer.solve_heuristic(e)
    print("  [INFO] Border final cost : {}".format(nb_conflict))
    initial_solution = generate_random_inner_solution(e, initial_solution)
    if debug: visualize(e, initial_solution, "debug/debuging_init")

    return lns(e, initial_solution, search_time=lns_search_time, neighborhood_size=neighborhood_size, allow_adjacent=allow_adjacent, tabu_queue_size=tabu_queue_size, debug=debug)



def lns(e: EternityPuzzle, solution, search_time=30, neighborhood_size=5, allow_adjacent=False, tabu_queue_size=0, debug=False):

    assert tabu_queue_size >= 0, "tabu_queue need to be greater of equal at 0"
    if tabu_queue_size > (limit_size:=(4 * e.board_size - 2) // 2):
        print(f"Tabu queue too big, setting it to half the number of inner piece: {limit_size}")
        tabu_queue_size = limit_size

    if debug: visualize(e, solution, "debug/before_destruction")
    start_time = time.time()
    tic = start_time
    n_conflit_best_solution = e.get_total_n_conflict(solution)

    tabu_queue = []

    print(f"  [INFO] Solving inner puzzle ...")
    with tqdm(total=search_time) as progress_bar:

        while True:

            new_solution = rebuild(*destroy(e, solution, neighborhood_size=neighborhood_size, tabu_idx=tabu_queue, allow_adjacent=allow_adjacent, debug_visualization=debug))
            if (nb_confllict_new_sol := e.get_total_n_conflict(new_solution)) < e.get_total_n_conflict(solution):
                solution = new_solution
                n_conflit_best_solution = nb_confllict_new_sol         
            
            new_solution = local_swap(e, solution)

            if (nb_confllict_new_sol := e.get_total_n_conflict(new_solution)) < e.get_total_n_conflict(solution):
                solution = new_solution
                n_conflit_best_solution = nb_confllict_new_sol  

            while len(tabu_queue) > tabu_queue_size:
                tabu_queue.pop(0)

            if (tac := time.time()) - start_time < search_time:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                break

    return solution, n_conflit_best_solution


def local_swap(e, solution):
    return gloton_place_removed_pieces(*destroy(e, solution, neighborhood_size=4, allow_adjacent=True))


def destroy(e: EternityPuzzle, solution, neighborhood_size: int=4, tabu_idx: List=None, allow_adjacent: bool=False, debug_visualization: bool=False):

    solution = deepcopy(solution) # To make sure we are not affecting the input solution

    # Remove k pieces with the most conflict
    conflict_position, idx_to_nb_conflict = get_conflict_positions(e, solution, return_nb_conflict=True)

    # Sample the distribution
    removed_pieces_idx = []
    while len(removed_pieces_idx) != neighborhood_size:
        last_removed_idx = removed_pieces_idx[-1] if len(removed_pieces_idx) > 0 else None
        priotize_adjacent = True if last_removed_idx != None else False
        idx_conflic, probabilities = get_probability_of_removing_piece(e, solution, idx_to_nb_conflict, priotize_adjacent=priotize_adjacent, last_removed_piece_idx=last_removed_idx)
        if len(idx_conflic) == 0: break
        idx = np.random.choice(idx_conflic, p=probabilities)
        # Check if it allow to remove the piece
        if not allow_adjacent and is_adjacent(e, idx, removed_pieces_idx): 
            del idx_to_nb_conflict[idx]
            continue
        if tabu_idx != None and idx in tabu_idx:
            del idx_to_nb_conflict[idx]
            continue
        removed_pieces_idx.append(idx)
        del idx_to_nb_conflict[idx]

    # Remove the selected pieces
    removed_pieces = []
    for idx in removed_pieces_idx:
        removed_pieces.append(solution[idx])
        if tabu_idx:
            tabu_idx.append(idx)
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


def get_probability_of_removing_piece(e: EternityPuzzle, solution, idx_to_nb_conflict: Dict, priotize_adjacent=False, last_removed_piece_idx=None):
    """ Find the probability of removing a piece base on the number of conflicts """
    if priotize_adjacent:
        for adj_idx in get_adjacent_idx(e, last_removed_piece_idx):
            if adj_idx in idx_to_nb_conflict.keys():
                idx_to_nb_conflict[adj_idx] += PRIORITY_WEIGHT

    pieces_with_most_conflict = [
        (idx, idx_to_nb_conflict[idx] + 0.5) for idx in idx_to_nb_conflict.keys() if piece_type(solution[idx]) == INNER
        ]
    
    idx_conflic = [idx for idx, _ in pieces_with_most_conflict]
    probabilities = np.array([nb_conflict for _, nb_conflict in pieces_with_most_conflict])
    probabilities = probabilities / np.sum(probabilities)
    return idx_conflic, probabilities


def is_adjacent(e: EternityPuzzle, idx: int, puzzle_pieces_idx: List[int]):
    idx_north, idx_east, idx_south, idx_west = get_adjacent_idx(e, idx)
    for piece_idx in puzzle_pieces_idx:
        if idx_north == piece_idx: return True
        if idx_south == piece_idx: return True
        if idx_east == piece_idx: return True
        if idx_west == piece_idx: return True

    return False
