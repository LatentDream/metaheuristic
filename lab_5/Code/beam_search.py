import heapq
import local_search as ls
import time as t

def generate_successor(atelier, population):
    """Generates all possible successor (assigning a currently machine currently unoccupied site)
    :param atelier: an instance of the problem
    :param population: a list of solutions of which to generate the successors
    
    :return: a list of successors solutions 
    (a solution is  a dictionnary where the keys are the machines and the values are the sites of the machines)
    """
    succ = []
    for solution in population:
        possible = [i for i in solution if solution[i]==0]
        sites = [j for j in range(atelier.n_machines) if j not in solution.values()]
        for m in possible:
            for j in sites:
                solution[m]=j
                succ.append(dict(solution))
                solution[m]=0
    return succ

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

def beam_search(atelier,args):
    """Builts an initial solution using a Beam Search
    :param atelier: an instance of the problem
    
    args
    :param pop_size: the size of the population
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    # Initializing the population
    population = [{i:0 for i in range(atelier.n_machines)}]

    for _ in range(atelier.n_machines-1):

        # Generating successors
        candidates = generate_successor(atelier, population)

        # Selecting the best ones
        population = heapq.nsmallest(args.pop_size, candidates, key=lambda a:atelier.get_total_cost(a))

    solution = local_search(atelier, min(population, key=lambda a:atelier.get_total_cost(a)), args)
    return solution