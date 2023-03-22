import time as t
import random as r
import heapq
import numpy as np

def generate_swap2_pop(atelier, population, args):
    """Generates a list of solutions by making a subset of possible permutation of 2 machines on each solution of the population
    :param atelier: an instance of the problem
    :param population: a list of dictionnaries where the keys are the machines and the values are the sites of the machines

    args
    :param p: the proportion of possible permutations considered
    
    :return: the list of all solutions generated
    """

    # Initialisation of return
    res = population[::]

    for solution in population:
        # Permutations to consider
        to_consider = r.sample([(i,j) for i in range(atelier.n_machines) for j in range(i+1,atelier.n_machines)], int(args.p*atelier.n_machines*(atelier.n_machines-1)))
        for (i,j) in to_consider:
            solution[i],solution[j]=solution[j],solution[i] # Computing permutation
            res.append(dict(solution))
            solution[i],solution[j]=solution[j],solution[i] # Coming back to original configuration
    return res

def local_beam_search(atelier, args):
    """Computes a local beamsearch
    :param atelier: an instance of the problem
    
    args
    :param pop_size: size of the population considered
    :param max_time: maximum computation time
    :param max_iter: maximum number of non-improving iterations

    :return: a dictionnary where the keys are the machines and the values are the sites of the machines
    """

    # Initialisation of return parameters

    t0 = t.time()
    n_iter = 0

    best_cost = 10**100
    best_sol = dict()

    # Generation of initial population
    population = [atelier.generate_initial_solution(args) for k in range(args.pop_size)]

    while t.time()-t0 < args.max_time and n_iter < args.max_iter:
        # Generating the candidates
        candidates = generate_swap2_pop(atelier, population, args)

        # Getting the k best among them
        population = heapq.nsmallest(args.pop_size, candidates, key=lambda a:atelier.get_total_cost(a))
        
        # Getting the best solution and updating return
        solution = min(population, key=lambda a:atelier.get_total_cost(a))
        cost = atelier.get_total_cost(solution)

        if cost < best_cost:
            best_cost = cost
            best_sol = solution
            n_iter=0
        else:
            n_iter+=1
    
    return best_sol
