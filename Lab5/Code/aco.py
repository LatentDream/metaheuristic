import numpy as np
import local_search as ls
import time as t
import heapq
import random as r

def construction(atelier, pher, args):
    """Computes the construction of a Random Ant Solution
    :param atelier: an instance of the problem
    :param pher: a matrix containing the values of the pheromones
    
    args
    :param phw: weight of pheromones in selection
    :param hew: weight of heuristic for the selection
    
    :return: a a dictionnary where the keys are the machines and the values are the sites of the machines"""
    
    # Initialization of return variables
    solution = {i:0 for i in range(atelier.n_machines)}
    curr_cost = 0

    sites = list(range(1,atelier.n_machines))
    r.shuffle(sites)

    for j in sites:
        curr_cost = atelier.get_total_cost(solution)
        worst_cost = curr_cost
        # Possible machines to assign to this site
        possible = [i for i in solution if solution[i]==0]

        # Initializing probabilities
        probas = dict()

        for v in possible:

            # Computing decision cost
            solution[v] = j
            cost = atelier.get_total_cost(solution)
            solution[v] = 0

            probas[v] = cost-curr_cost

            # Updating reference for this decision
            if cost > worst_cost:
                worst_cost = cost
        
        # Computing probabilities
        for p in probas:
            probas[p] = pher[j,p]**args.phw*(p / worst_cost)**args.hew
        
        # Normalizing
        total = sum(probas.values())
        for p in probas:
            probas[p] = probas[p]/total

        # Getting cumulative probabilities
        new_probas = {p:0 for p in probas}
        order = list(new_probas.keys())
        r.shuffle(order)

        for p in order:
            new_probas[p]=probas[p]+max(new_probas.values())
        
        # Choosing the machine to put on the site
        a = r.random()
        m = min([(k,v) for (k,v) in new_probas.items() if v>a], key=lambda a:a[1])[0]
        solution[m] = j

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


def update(atelier, population, pher, args):
    """Computes an update of the pheromones
    :param atelier: an instance of the problem
    :param population: the population of solution generated
    :param pher: a matrix containing the values of the pheromones
    
    args
    :param ph_evap: evaporation factor of the pheromones
    :param ph_rein: proportion of the solution which will see their associated pheromones reinforced
    :param ph_max: maximum value of the pheromones
    :param ph_min: minimum value of the pheromones
    
    :return: a matrix containing the values of the pheromones
    """

    # Evaporation step
    for ph in pher:
        ph *= (1-args.ph_evap)

    # Getting solutions to reinforce
    reinforced = heapq.nsmallest(int(args.ph_rein*args.n_ants), population, key=lambda a:atelier.get_total_cost(a))

    # The reinforcement will be equal to the worst cost of the population divided by the cost of the solution considered
    ref = max([atelier.get_total_cost(a) for a in population])
    for solution in reinforced:
        cost = atelier.get_total_cost(solution)
        for i in solution:
            pher[i,solution[i]] += ref/cost

    for j in pher:
        for i in j:
            if i < args.ph_min:
                i = args.ph_min
            elif i > args.ph_max:
                i = args.ph_max

    return pher

def aco(atelier, args):
    """Computes an Ant Coloniy Optimization
    :param atelier: an instance of the problem
    
    args
    :param n_ants: number of ants in the population
    :param max_time: maximum computation time
    :param n_travel: number of travels of the ants (= number of iterations)
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines
    """
    
    # Initialising return variables
    t0 = t.time()
    n_trav = 0

    best_cost = 10**10
    best_sol = dict()

    # Initialising pheromones
    pher = np.ones((atelier.n_machines, atelier.n_machines))
    for i in pher:
        i = args.ph_max

    while n_trav < args.n_travel and t.time()-t0 < args.max_time:
        
        # Population of solution at each travel
        population = []

        for _ in range(args.n_ants):
            solution = construction(atelier, pher, args)
            solution = local_search(atelier, solution, args)
            population.append(solution)
            
            cost = atelier.get_total_cost(solution)
            if cost < best_cost:
                best_cost = cost
                best_sol = solution
        
        pher = update(atelier, population, pher, args)
        n_trav += 1
    
    return best_sol