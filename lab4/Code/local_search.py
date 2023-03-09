import random as r
import time as t

def swap2(atelier, solution, args):
    """Computes the best permutation possible among a subset of possible permutations of two machines
    :param atelier: an instance of the problem
    :param solution: a dictionnary where the keys are the machines and the values are the sites of the machines

    args
    :param p: the proportion of possible permutations considered
    
    :return: a tuple containing the cost and the tuple containg the two machines concerned by the permutation chosen
    """

    # Permutations to consider
    to_consider = r.sample([(i,j) for i in range(atelier.n_machines) for j in range(i,atelier.n_machines) if i!=j], int(0.5*args.p*atelier.n_machines*(atelier.n_machines-1)))
    
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

    return best_cost, best_perm

def local_search(atelier, args):
    """Computes a 2-swap based local search
    :param atelier: an instance of the problem

    args
    :param max_iter: maximum number of non-improving iterations
    :param max_time: maximum computation time
    :param mode: decides whether the initial solution is random or greedy
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines
    """

    # Initialisation
    solution = atelier.generate_initial_solution(args)

    best_cost = atelier.get_total_cost(solution)
    n_iter = 0
    t0 = t.time()

    while n_iter < args.max_iter and t.time()-t0 < args.max_time:
        cost, perm = swap2(atelier, solution, args)     
        
        # Update solution if there is an amelioration and number of non-improving iteration
        if cost < best_cost:                            
            n_iter = 0                                  
            best_cost = cost
            solution[perm[0]],solution[perm[1]] = solution[perm[1]],solution[perm[0]]
        else:
            n_iter += 1

    return solution

def restarts(atelier, args):
    """Computes a local search using restarts for diversification
    :param atelier: an instance of the problem
    
    args
    :param max_time: maximum computation time allowed
    :param max_restarts: maximum number of restarts allowed
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    # Initialisation
    best_solution = dict()
    best_cost = 10**10

    t0 = t.time()
    n_restarts = 0

    while n_restarts < args.max_restarts and t.time()-t0 < args.max_time:
        solution = local_search(atelier, args)  
        cost = atelier.get_total_cost(solution)

        # Update if amelioration
        if cost < best_cost:
            best_cost = cost
            best_solution = dict(solution)
        n_restarts += 1
    
    return best_solution