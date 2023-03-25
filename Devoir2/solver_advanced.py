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
    # nb_of_iter = 100_000    # Stopping criteria 
    nb_of_iter = 1050        # Stopping criteria 
    time_limit = 20*60      # Stopping criteria 
    nb_of_ants = 1          # n_of_ants: the number of ants
    l_rate = 0.1            # l_rate: the learning rate for pheromone values
    tau_min = 0.001         # lower limit for the pheromone values
    tau_max = 0.999         # upper limit for the pheromone values
    determinism_rate = 0.1  # rate of determinism in the solution construction
    nb_of_trials = 4        #  number of trials to be executed for the given problem instance
    beam_width = 10          # parameters for the beam procedure
    mu = 5.0                # stochastic sampling parameter
    max_children = 100      # stochastic sampling parameter #! NOT USED 
    n_samples = 5           # stochastic sampling parameter
    sample_percent = 100    # stochastic sampling parameter #! NOT USED 
    do_local_search = True  # If the local search heuristic is executed
    
    ant = Ant(tsptw, l_rate=l_rate, tau_max=tau_max, tau_min=tau_min)
    pbs = ProbabilisticBeamSearch(tsptw, ant, beam_width, determinism_rate, max_children, mu, n_samples, sample_percent)

    # To collec statistics
    best_solution = None
    results = list()
    violations = list()
    times_best_found = list()
    iter_best_found = list() 

    # Solution
    best_so_far_solution = None
    restart_best_solution = None
    iteration_best_solution = None

    ### For each trial for a number of trial
    with tqdm(total=nb_of_iter*nb_of_trials) as progress_bar:

        for trial_nb in range(nb_of_trials):
            #? Beam-ACO: algo #2 of the paper
            best_so_far_solution = None
            restart_best_solution = None
            ant.resetUniformPheromoneValues()
            bs_update = False
            restart = False

            ### algorithm: iterates main loop until a maximum CPU time limit is reached
            trial_tic = time.time()
            time_init = time.time()
            nb_iter_done = 0
            while nb_iter_done < nb_of_iter:

                if time.time() - time_init  > time_limit:
                    progress_bar.update(nb_of_iter - nb_iter_done)
                    break

                iteration_best_solution = None

                ### Probabilistic beam search algorithm is executed. This produces the iteration-best solution Pib
                for i in range(nb_of_ants):
                    #?  Probabilistic beam search: This part is the algo #1 of the paper
                    iteration_best_solution = pbs.beam_construct()
                    ### Then subject to the application of local search
                    if do_local_search:
                        new_solution = local_search(iteration_best_solution, tsptw)
                        iteration_best_solution = get_best_soltion(new_solution, iteration_best_solution, tsptw)
                
                
                ### Updating the best-so-far solution
                trial_tac = time.time()
                if restart:
                    restart = False
                    restart_best_solution = None
                    best_so_far_solution = get_best_soltion(best_so_far_solution, iteration_best_solution, tsptw)    
                else:
                    restart_best_solution = get_best_soltion(iteration_best_solution, restart_best_solution, tsptw)
                    best_so_far_solution = get_best_soltion(best_so_far_solution, iteration_best_solution, tsptw)
                    
                best_soltion = get_best_soltion(best_solution, best_so_far_solution, tsptw)
                
                # Stats
                results.append(get_score(best_so_far_solution, tsptw))
                violations.append(get_number_of_violations(best_so_far_solution, tsptw))
                times_best_found.append(trial_tic-trial_tac)
                iter_best_found.append(trial_nb)
                
                
                ### A new value for the convergence factor cf is computed
                cf = ant.computeConvergenceFactor()

                
                ### Depending on cf and bs_update, a decision on whether to restart the algorithm or not is made
                if bs_update and cf > 0.99:
                    ant.resetUniformPheromoneValues()
                    bs_update = False
                    restart = True
                else:
                    if cf > 0.99:
                        bs_update = True
                    ant.updatePheromoneValues(bs_update, cf, iteration_best_solution, restart_best_solution, best_so_far_solution)
                
                trial_tic = time.time()
                nb_iter_done += 1
                progress_bar.update(1)
    
    print(f"results: {zip(results, violations)} \n")
    save_stats_as_fig(results, violations, times_best_found, iter_best_found)
    print(f"Number of violation: {get_number_of_violations(best_soltion, tsptw)}")
    print(f"Feasible solution: {tsptw.verify_solution(best_soltion)}")
    return best_soltion
