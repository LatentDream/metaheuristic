import matplotlib.pyplot as plt
from typing import Dict, List
from utils.vns_utils import prioritise_neighborhhod, swap
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
    vns_search_time = 1 * MINUTE

    random.seed(1998)

    select_random_neighborhood(e, debug=True)
    raise Exception("END PROG")


    # Build an inital solution with good border
    if info: print(f"  [INFO] Solving border puzzle ...")
    initial_solution, nb_conflict = solver_heuristic_layer.solve_heuristic(e)
    print("  [INFO] Border final cost : {}".format(nb_conflict))
    initial_solution = generate_random_inner_solution(e, initial_solution)
    if debug: visualize(e, initial_solution, "debug/debuging_init")


    # Solve inner puzzle
    return multi_start_vns(e, initial_solution, search_time=vns_search_time, info=info, debug=debug)


def multi_start_vns(e: EternityPuzzle, initial_solution, search_time=30, info=False, debug=False):

    if info: print(f"  [INFO] Solving inner puzzle ...")

    solution = vnd(initial_solution)

    # VNS parameters
    p = 0
    LEVEL_MAX = e.board_size
    ITER_MAX = 200
    stop = False

    # 
    start_time = time.time()
    tic = start_time
    with tqdm(total=search_time) as progress_bar:
        while p < LEVEL_MAX and not stop:
            iteration_number = 0
            while iteration_number < ITER_MAX and not stop:
                new_solution == copy.deepcopy(solution)
                for _ in range(0, p + 2):
                    k = select_random_neighborhood(e)
                    new_solution = shake(new_solution, k)
                optimised_new_solution = vnd(new_solution)
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
    return solution



def vnd(e: EternityPuzzle, solution, swap_per_iter=50, nb_neigborhood=10, debug=False):

    # Start from the border and go to the to the center slowly
    for _ in range(0, nb_neigborhood):
        # Select a neigborhood
        k = np.ones(e.board_size, e.board_size) # Todo
        # Local swap on that neigborhood
        for _ in range(0, swap_per_iter):
            solution = swap(e, solution, k, debug=debug) # Two swap
            solution = swap(e, solution, k, nb_piece_to_swap=1, debug=debug) # Rotate a piece

    return solution

    
def select_random_neighborhood(e: EternityPuzzle, debug=False):
    i, j = int(np.random.uniform(0, e.board_size)), int(np.random.uniform(0, e.board_size))
    neighborhood = prioritise_neighborhhod(e, i, j, sigma=1, inverted=False, debug=False)

    return neighborhood


def shake(solution, k):
    """ Swap some random piece to destroy a bit the solution """
    raise NotImplementedError()
