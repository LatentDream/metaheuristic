from typing import List, Tuple
from tqdm import tqdm
from tsptw import TSPTW
from utils.ant import Ant
from utils.beam_search import ProbabilisticBeamSearch
import time

from utils.local_search import get_number_of_violations, local_search
from utils.utils import save_stats_as_fig, get_best_soltion, get_score


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        tsptw (TSPTW): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation 
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    # Variables
    # nb_of_iter = 100_000   # stopping criteria 
    nb_of_iter = 1           # stopping criteria 
    determinism_rate = 0.01  # rate of determinism in the solution construction
    nb_of_trials = 10        # number of trials to be executed for the given problem instance
    beam_width = 50          # parameters for the beam procedure
    mu = 100.0               # stochastic sampling parameter
    
    pbs = ProbabilisticBeamSearch(tsptw, None, beam_width, determinism_rate, mu)

    # To collec statistics
    best_solution = None
    results = list()
    violations = list()
    times_best_found = list()
    iter_best_found = list()

    total_iteration = nb_of_iter * nb_of_trials if nb_of_iter != 0 else nb_of_trials
    with tqdm(total=total_iteration) as progress_bar:
        for trial_nb in range(nb_of_trials):
            trial_tic = time.time()
            
            trial_solution = pbs.beam_construct()
            trial_best_solution = trial_solution
            best_solution = get_best_soltion(best_solution, trial_solution, tsptw)

            for _ in range(5):
                trial_solution = local_search(trial_solution, tsptw)
                trial_best_solution = get_best_soltion(trial_solution, trial_best_solution, tsptw)  
                    
            trial_tac = time.time()

            best_solution = get_best_soltion(best_solution, trial_solution, tsptw)
                    
            # Stats
            results.append(get_score(trial_solution, tsptw))
            violations.append(get_number_of_violations(trial_solution, tsptw))
            times_best_found.append(trial_tic-trial_tac)
            iter_best_found.append(trial_nb)
                    
            trial_tic = time.time()
            progress_bar.update(1)
    
    print(f"results: {zip(results, violations)} \n")
    save_stats_as_fig(results, violations, times_best_found, iter_best_found)
    print(f"Number of violation: {get_number_of_violations(best_solution, tsptw)}")
    print(f"Feasible solution: {tsptw.verify_solution(best_solution)}")
    return best_solution
