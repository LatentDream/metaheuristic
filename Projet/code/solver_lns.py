import numpy as np
import math
import copy
from solver_genetic import genetic_algorithm_border
from utils.utils import *
import itertools
from copy import deepcopy
import time
from tqdm import tqdm

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
    visualize(e, initial_solution, "debogging_border")


    return lns(e, initial_solution, search_time=1*60)

def lns(e: EternityPuzzle, solution, search_time=30):

    start_time = time.time()
    tic = start_time
    best_solution = deepcopy(solution)
    n_conflit_best_solution = e.get_total_n_conflict(best_solution)

    with tqdm(total=search_time) as progress_bar:

        while True:

            new_solution = rebuild(*destroy(e, solution))

            if nb_confllict_new_sol := (e.get_total_n_conflict(new_solution)) < e.get_total_n_conflict(solution):
                solution = new_solution
                if nb_confllict_new_sol < n_conflit_best_solution:
                    best_solution = deepcopy(new_solution)
                    n_conflit_best_solution = nb_confllict_new_sol

            if (tac := time()) - start_time < search_time:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                break

    return best_solution, n_conflit_best_solution


def destroy(e: EternityPuzzle, solution):
    solution = deepcopy(solution) # To make sure we are not affecting the input solution
    print(solution)
    visualize(e, solution, "debouging")
    print(get_conflict_positions(e, solution))
    # Remove pieces 


    return e, solution



def rebuild(e: EternityPuzzle, destroyed_solution):
    # Reallocated removed pieces optimally to the holes


    raise NotImplementedError()