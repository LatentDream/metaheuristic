import random as r
import local_search as ls
import time as t
from math import *

def local_search(atelier, solution, args):
    """Computes a 2-swap based local search
    :param atelier: an instance of the problem
    :param solution: an initial solution (a dictionnary where the keys are the machines and the values are the sites of the machines)

    args
    :param max_iter: maximum number of non-improving iterations
    :param max_time: maximum computation time
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines
    """

    # Initialisation
    best_cost = atelier.get_total_cost(solution)
    n_iter = 0
    t0 = t.time()

    while n_iter < args.max_iter and t.time()-t0 < args.max_time:
        cost, perm = ls.swap2(atelier, solution, args)     
        
        # Update solution if there is an amelioration and number of non-improving iteration
        if cost < best_cost:                            
            n_iter = 0                                  
            best_cost = cost
            solution[perm[0]],solution[perm[1]] = solution[perm[1]],solution[perm[0]]
        else:
            n_iter += 1

    return solution

def perturbation(atelier, solution, args):
    """Computes a perturbation on a solution by reversing the sites of some machines 
    (ex : {0:1, 1:2, 2:3, 3:4, 4:5} can become {0:1, 1:5, 2:4, 3:3, 4:2})
    :param atelier: an instance of the problem
    :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines
    
    args
    :param pert: proportion of the solution
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    i_min = r.randint(0,len(solution)-1)
    i_max = i_min + int(args.pert*len(solution))
    perturb = dict(solution)

    if i_max <= len(solution):
        slice = list(range(i_min,i_max))
        slice = slice[::-1]
    else:
        slice = list(range(i_min,len(solution))) + list(range(i_max-len(solution)))
        slice = slice[::-1]
    
    new_sol={slice[i]:solution[slice[-i-1]] for i in range(len(slice))}

    for i in new_sol:
        perturb[i] = new_sol[i]
    
    return perturb

def ils(atelier, args):
    """Computes an iterated local search
    :param atelier: an instance of the problem
    
    args
    :param max_time: maximum computation time
    :param accept_temp: initial acceptation temperature
    :param red_factor: reduction factor of acceptation temperature
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    # Initialization of return variable
    t0 = t.time()
    
    best_cost = 10**10
    best_sol = dict()
    
    #Initial solution

    temp = args.accept_temp
    solution = local_search(atelier, atelier.generate_initial_solution(args), args)
    curr_cost = atelier.get_total_cost(solution)
    curr_sol = dict(solution)

    while t.time()-t0 < args.max_time:
        
        # Getting a new solution
        solution = local_search(atelier,perturbation(atelier, curr_sol, args), args)
        cost = atelier.get_total_cost(solution)

        # Updating return variables
        if cost < best_cost:
            best_cost = cost
            best_sol = dict(solution)
            curr_cost = cost
            curr_sol = dict(solution)
        
        # Trying acceptation
        else:
            a = r.random()
            if a < exp((cost-curr_cost)/temp):
                curr_cost = cost
                curr_sol = solution

        # Updating acceptation temperature
        temp *= args.red_factor

    return best_sol