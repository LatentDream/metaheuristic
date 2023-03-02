from typing import List, Tuple
from tsptw import TSPTW
import time


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation 
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """

    ### 1. all variables are initialized. The pheromone values are set to their initial value 0.5

    nb_of_iter = 100,000    # Stopping criteria 
    time_limit = 30*60      # Stopping criteria 
    nb_of_ants = 1          # n_of_ants: the number of ants
    l_rate = 0.1            # l_rate: the learning rate for pheromone values
    tau_min = 0.001         # lower limit for the pheromone values
    tau_max = 0.999         # upper limit for the pheromone values
    determinism_rate = 0.9  # rate of determinism in the solution construction
    nb_of_trials = 1         #  number of trials to be executed for the given problem instance
    beam_width = 1          # parameters for the beam procedure
    mu = 2.0                # stochastic sampling parameter
    max_children = 100      # stochastic sampling parameter
    n_samples = 10          # stochastic sampling parameter
    sample_percent = 100    # stochastic sampling parameter
    sample_rate = -1        # stochastic sampling parameter

    trial_time = 30 * 60
    tic = time.time()

    # TODO: Init Ant

    to_choose = int(beam_width * mu)
    sample_rate = int((sample_percent * (tsptw.num_nodes - 1) / 100.0) + 0.5) + 1

    # collecting statistics
    best_solution = None
    results = list()
    viols = list()
    times_best_found = list()
    iter_best_found = list()

    ### Then, the algorithm iterates a main loop until a maximum CPU time limit is reached
    for trial_nb in range(nb_of_trials):
        pass

        ### A probabilistic beam search algorithm is executed. This produces the iteration-best solution Pib

        ### Then subject to the application of local search

        ### Updating the best-so-far solution

        ### A new value for the convergence factor cf is computed

        ### Depending on cf and bs_update, a decision on whether to restart the algorithm or not is made
    
    raise Exception("Advanced solver not implemented")


def Beam_ACO():
    # See paper
    raise Exception(f"{Beam_ACO.__name__} is not implemented")



def local_search():
    # based on the 1-opt neighborhood in which a single customer is 
    # removed from the tour and reinserted in a different position

    raise Exception(f"{local_search.__name__} is not implemented")