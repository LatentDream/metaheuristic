import matplotlib.pyplot as plt
from typing import Dict, List
from solver_genetic import genetic_algorithm_border
from utils.vns_utils import prioritise_neighborhhod, swap, choose
from utils.utils import *
from tqdm import tqdm
import solver_heuristic_layer
import numpy as np
import copy
import time



def solve_vns(e: EternityPuzzle):
    """
    Variable neighborhood search
    """
    #! Debug option
    debug = False
    info = True

    #? Parameter
    vns_search_time = 20 * MINUTE

    random.seed(1998)
    border_time = 1
    pop_size = 30
    mutation_rate = 0
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 10
    elite_size = 2

    # Build an inital solution with good border
    if info: print(f"  [INFO] Solving border puzzle ...")
    initial_solution, nb_conflict = genetic_algorithm_border(e, num_generations=num_generations, no_progress_generations=no_progress_generations, elite_size=elite_size, tournament_size=tournament_size, tournament_accepted=tournament_accepted, pop_size=pop_size, time_limit=border_time, debug_visualization=debug)
    # initial_solution, nb_conflict = solver_heuristic_layer.solve_heuristic(e)
    print("  [INFO] Border final cost : {}".format(nb_conflict))
    initial_solution = generate_random_inner_solution(e, initial_solution)
    if debug: visualize(e, initial_solution, "debug/debuging_init")


    # Solve inner puzzle
    return multi_start_vns(e, initial_solution, search_time=vns_search_time, info=info, debug=debug)


def multi_start_vns(e: EternityPuzzle, initial_solution, search_time=30, info=False, debug=False):

    if info: print(f"  [INFO] Solving inner puzzle ...")

    solution = vnd(e, initial_solution, debug=debug)

    # VNS parameters
    p = 0
    LEVEL_MAX = e.board_size
    ITER_MAX = 200
    stop = False

    start_time = time.time()
    tic = start_time
    with tqdm(total=search_time) as progress_bar:
        while p < LEVEL_MAX and not stop:
            iteration_number = 0
            while iteration_number < ITER_MAX and not stop:
                new_solution = copy.deepcopy(solution)
                for _ in range(0, p + 2):
                    k = select_random_neighborhood(e, debug=debug)
                    new_solution = shake(e, new_solution, k, debug=debug)
                optimised_new_solution = vnd(e, new_solution, 200, debug=debug)
                if e.get_total_n_conflict(optimised_new_solution) < e.get_total_n_conflict(solution):
                    solution = optimised_new_solution
                    p = 0
                    iteration_number = 0
                iteration_number += 1
                if (tac := time.time()) - start_time < search_time:
                    progress_bar.update(tac - tic)
                    tic = tac
                else:
                    stop = True
            p += 1
    return solution, e.get_total_n_conflict(solution)



def vnd(e: EternityPuzzle, solution, swap_per_iter=100, debug=False):
    # Start from the border and go to the to the center slowly
    sigma_neighborhood=[(1.8, 1, 0.15), (1.5, 0.8, 0.1), (1.5, 1.0, 0.05), (None, 0.2, None)]

    for s_outer, s, s_inner in sigma_neighborhood:
        # Buid the neighborhood by removing the gaussian s_outer and s_inner to s
        i, j = e.board_size // 2,  e.board_size // 2
        k = prioritise_neighborhhod(e, i, j, sigma=s)
        if s_inner:
            k_inner = prioritise_neighborhhod(e, i, j, sigma=s_inner)
            k = k - k_inner
            k[k < 0] = 0
            k = k / np.sum(k)
        if s_outer:
            k_outer = prioritise_neighborhhod(e, i, j, sigma=s_outer, inverted=True)
            k = 1 + (k - k_outer)
            k = k / np.sum(k)
        k = k / np.sum(k)
        if debug:
            plt.imshow(k, cmap='binary')
            plt.colorbar()
            plt.savefig("debug/selected_neighborhood")
            plt.close()

        # Local swap on the selected neighborhood k
        for _ in range(0, swap_per_iter):
            pass
            solution = swap(e, solution, k.flatten()) # Two swap
            solution = swap(e, solution, k.flatten(), nb_piece_to_swap=1) # Rotate a piece

        if debug: visualize(e, solution, "debug/vns")
        

    return solution

    
def select_random_neighborhood(e: EternityPuzzle, debug=False):
    i, j = int(np.random.uniform(0, e.board_size)), int(np.random.uniform(0, e.board_size))
    neighborhood = prioritise_neighborhhod(e, i, j, sigma=1, inverted=False, debug=debug)

    return neighborhood


def shake(e: EternityPuzzle, solution, probabilities, nb_piece=-1, debug=False):
    """ Swap some random piece to destroy a bit the solution """
    if nb_piece == -1:
        nb_piece = e.board_size

    # Select nb_piece
    probabilities = probabilities / np.sum(probabilities)
    _, destroyed_solution, removed_pieces, holes_idx, _ = choose(e, solution, probabilities.flatten(), nb_piece, debug=debug)

    # Replace the pieces randomdly
    i = 0
    while i < len(holes_idx):
        hole_idx = holes_idx[i]
        destroyed_solution[hole_idx] = removed_pieces[i]
        i += 1

    if debug: visualize(e, destroyed_solution, 'debug/after_shaking')

    return destroyed_solution
