import local_search as ls
import random as r
import time as t

def construction(atelier,args):
    """Computes a greedy randomized construction of an initial solution for GRASP
    :param atelier: an instance of the problem
    
    args
    :param rand_fact: randomization factor

    :return: the solution generated
    """
    # Initialisation
    solution = dict()
    for i in range(atelier.n_machines):
        solution[i] = 0
    
    # Randomly determining the orders of the sites
    sites = list(range(1,atelier.n_machines))
    r.shuffle(sites)

    for j in sites:
        possible = [a for a in range(atelier.n_machines) if solution[a]==0]             # Machines still to assign
        costs=dict()
        for i in possible:
            solution[i] = j
            costs[i]=atelier.get_total_cost(solution)
            solution[i] = 0
        
        h = min(costs.values()) + args.rand_fact*(max(costs.values())-min(costs.values()))  # Upper bound to choose next assignation
        solution[r.sample([i for i in possible if costs[i]<=h],1)[0]] = j

    return solution

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

def grasp(atelier,args):
    """Computes a greedy randomized adaptative search procedure
    :param atelier: an instance of the problem
    
    args
    :max_time: maximum computation time
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    t0 = t.time()

    best_sol = dict()
    best_cost = 10**10

    while t.time()-t0 < args.max_time:
        solution = construction(atelier,args)
        solution = local_search(atelier,solution,args)
        cost = atelier.get_total_cost(solution)
        if cost < best_cost:
            best_sol = solution
            best_cost = cost
    
    return best_sol