from atelier import Atelier
import local_search as ls
import time as t
import random as r
from math import *

def simulated_annealing(atelier, args):
    """Computes a simulated annealing using 2-swap local search
    :param atelier: an instance of the problem
    
    args
    :param mode: decides whether the initial solution is random or greedy
    :param max_iter: maximum number of non-improving iterations
    :param init_temp: initial temperature
    :param red_factor: reduction factor of the temperature at each iteration
    :param max_time: maximum computation time allowed
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    # Initialisation
    temperature = args.init_temp

    t0 = t.time()
    n_iter = 0

    solution = atelier.generate_initial_solution(args)
    curr_cost = atelier.get_total_cost(solution)
    
    best_cost = curr_cost
    best_solution = dict(solution)

    while n_iter < args.max_iter and t.time()-t0 < args.max_time:
        cost, perm = ls.swap2(atelier, solution, args)
        
        # Always accept an improving neighbour
        if cost < curr_cost:
            n_iter = 0          # Update number of non-improving iteration
            curr_cost = cost
            solution[perm[0]],solution[perm[1]] = solution[perm[1]],solution[perm[0]]

            if cost < best_cost:
                best_cost = cost
                best_solution = dict(solution)
        
        # Else, use degradation acceptation probability
        elif cost > curr_cost:
            p = r.random()
            if p < exp((curr_cost-cost)/temperature):
                curr_cost = cost
                solution[perm[0]],solution[perm[1]] = solution[perm[1]],solution[perm[0]]
            
            n_iter+=1                   # Update number of non-improving iteration
            
        temperature *= args.red_factor  # Update Temperature

    return best_solution