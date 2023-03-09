from atelier import Atelier
import time as t
import random as r
import numpy as np

def swap2(atelier, solution, tabu, args):
    """Computes the best permutation possible among a subset of possible permutations of two machines
    :param atelier: an instance of the problem
    :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines
    :param tabu: a list of tuples any of each contains a tabu permutation

    args
    :param p: the proportion of possible permutations considered
    
    :return: a tuple containing the cost and the tuple containg the two machines concerned by the permutation chosen
    """

    # Permutations to consider
    to_consider = r.sample([(i,j) for i in range(atelier.n_machines) for j in range(i,atelier.n_machines) if i!=j], int(0.5*args.p*atelier.n_machines*(atelier.n_machines-1)))
    to_consider = [(i,j) for (i,j) in to_consider if (i,j) not in tabu and (j,i) not in tabu]

    # Initialisation of return
    best_cost = 10**10
    best_perm = (0,0)

    for (i,j) in to_consider:
        solution[i],solution[j] = solution[j],solution[i]   # Computes each permutation
        cost = atelier.get_total_cost(solution)
        if cost < best_cost:                                # Checking amelioration
            best_cost = cost                                # updating return
            best_perm = (i,j)
        solution[i],solution[j] = solution[j],solution[i]   # Coming back to initial solution

    solution[best_perm[0]],solution[best_perm[1]] = solution[best_perm[1]],solution[best_perm[0]] # Computing the best permutation found
    
    return best_cost, best_perm

def tabu_search(atelier, args):
    """ Computes a tabu search
    :param atelier: an instance of the problem

    args
    :param max_iter: maximum number of non-improving iterations
    :param max_time: maximum computation time allowed
    :param mu:, :param sigma: parameters of standard distribution used to update the tabu list

    :return: a dictionnary where the keys are the machines and the values are the sites of the machines
    """

    #Initialisation

    t0 = t.time()
    n_iter = 0
    tabu = dict()

    solution = atelier.generate_initial_solution(args)
    curr_cost = atelier.get_total_cost(solution)
    
    best_cost = curr_cost
    best_solution = dict(solution)

    while n_iter < args.max_iter and t.time()-t0 < args.max_time:
        cost, perm = swap2(atelier, solution, tabu, args)
        
        # Updating best known solution and number of non-improving iteration
        if cost < curr_cost:
            n_iter = 0
            if cost < best_cost:
                best_cost = cost
                best_solution = dict(solution)
        else:
            n_iter += 1
        
        # Updating tabu list
        for key in dict(tabu):
            tabu[key] = tabu[key]-1
            if tabu[key] == 0:
                del tabu[key]

        tabu[perm] = max(1,int(args.mu + args.sigma*np.random.randn()))
    
    return best_solution
        