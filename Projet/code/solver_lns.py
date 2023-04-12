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

    random.seed(1998)

    # Solve the border
    border_time = 1
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
    )

    print("Border final cost : {}".format(nb_conflict))
    
    initial_solution = generate_random_solution(e)
    visualize(e, initial_solution, "debogging_border")

    

    return lns(e, initial_solution, search_time=1*60)

def lns(e: EternityPuzzle, solution, search_time=30):

    visualize(e, solution, "debug/before_destruction")
    start_time = time.time()
    tic = start_time
    best_solution = deepcopy(solution)
    n_conflit_best_solution = e.get_total_n_conflict(best_solution)

    with tqdm(total=search_time) as progress_bar:

        while True:

            new_solution = rebuild(*destroy(e, solution, neighborhood_size=4, debug_visualization=True))

            if (nb_confllict_new_sol := e.get_total_n_conflict(new_solution)) < e.get_total_n_conflict(solution):
                solution = new_solution
                print(nb_confllict_new_sol)
                if nb_confllict_new_sol < n_conflit_best_solution:
                    best_solution = deepcopy(new_solution)
                    n_conflit_best_solution = nb_confllict_new_sol

            if (tac := time.time()) - start_time < search_time:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                break

    return best_solution, n_conflit_best_solution


def destroy(e: EternityPuzzle, solution, neighborhood_size=4, debug_visualization=False):

    solution = deepcopy(solution) # To make sure we are not affecting the input solution

    # Remove k pieces with the most conflict
    conflict_position, nb_conflict = get_conflict_positions(e, solution, return_nb_conflict=True)
    pieces_with_most_conflict = sorted([(idx, nb_conflict[idx]) for idx in nb_conflict.keys()], key= lambda x: -x[1])

    # p = np.array([for ])

    removed_pieces_idx = []
    while len(removed_pieces_idx) != neighborhood_size and len(pieces_with_most_conflict) != 0:
        piece_loc_idx = pieces_with_most_conflict.pop(0)[0]
        if piece_type(solution[piece_loc_idx]) == INNER:
            removed_pieces_idx.append(piece_loc_idx)
        if np.random.uniform(0.0, 1.0) < 0.25:
            pieces_with_most_conflict.pop(0)
        

    removed_pieces = []
    for idx in removed_pieces_idx:
        removed_pieces.append(solution[idx])
        solution[idx] = (BLACK, BLACK, BLACK, BLACK)

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
        visualize(e, destroyed_solution, "debug/rebuild")

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

