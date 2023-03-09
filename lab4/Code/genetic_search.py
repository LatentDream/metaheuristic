import time as t
import random as r

def generate_population(atelier, args):
    """Generates an initial population
    :param atelier: an instance of the problem
    
    args
    :param pop_size: size of population

    :return: a list of solution (a solution is a dictionnary where the keys are the machines and the values are the sites of the machines)"""

    return [atelier.generate_greedy_solution() for _ in range(args.pop_size)]

def roulette(atelier, population):
    """Computes a random weighted sample of half the individuals using cost as weight
    :param atelier: an instance of the problem
    :param population: the population to sample

    :return: the list of individuals selected
    """

    assert len(population) % 2 == 0, "On veut une population de taille paire"
    
    # List of costs od the solutions
    costs = []
    for sol in population:
        costs.append(atelier.get_total_cost(sol))

    # Initialisation of selected individuals
    selected = []

    while len(selected) != len(population):
        total_cost = sum(costs)

        # Determining the cumulative probabilities of selection
        probas = []
        curr = 0
        for cost in costs:
            curr += cost/total_cost
            probas.append(curr)

        # Choosing an individual
        a = r.random()
        i = 0
        while probas[i] < a:
            i += 1
        
        # Updating selected individuals, population and costs
        selected.append(population.pop(i))
        costs.pop(i)

    return selected

def crossing(atelier, population):
    """Computes a uniform crossing operation between the individuals of the population
    :param atelier: an instance of the problem
    :param population: the population to cross

    :return: the list of individuals created"""
    
    assert len(population)%2 == 0, "Population de taille paire requise"
    r.shuffle(population)
    
    for i in range(len(population)//2):

        # Initialisation of parents
        p1 = population[2*i]
        p2 = population[2*i+1]

        # Creation of Uniform crossing vector
        ux = [r.randint(0,1) for i in range(atelier.n_machines)]
        
        # Initialisation of children
        c1 = dict()
        c2 = dict()
        i1 = [] # Useful to complete the children 
        i2 = [] # in a feasible way

        for a in range(len(ux)):
            
            # Copying the elements corresponding to a 1 in ux from parents to children
            if ux[a] == 1:
                c1[a] = p1[a]
                c2[a] = p2[a]
            else:
                c1[a] = -1
                i1.append(a)
                c2[a] = -1
                i2.append(a)
        
        # Completing the children using the same order of apparition of the sites than in the other parent
        for j in range(len(p1)):
            if p1[j] not in c2.values():
                c2[i2.pop(0)] = p1[j]
            if p2[j] not in c1.values():
                c1[i1.pop(0)] = p2[j]
        
        population.append(c1)
        population.append(c2)
    
    return population

def mutation(atelier, population, args):
    """Computes a mutation of the individuals of the population
    :param atelier: an instance of the problem
    :param population: the population to mutate

    args
    :param mut_rate: probability for a child to endure a mutation

    :return: the list of individuals after mutation"""

    for sol in population:
        a = r.random()

        # Check if mutation has to be done
        if a < args.mut_rate:

            # Computes a sequence of simple permutations
            bef = r.sample(list(range(atelier.n_machines)),int(atelier.n_machines*0.1))
            mem = sol[bef[0]]
            for i in range(len(bef)-1):
                sol[bef[i]] = sol[bef[i+1]]
            sol[bef[-1]]=mem
    return population

def genetic_search(atelier, args):
    """Computes a genetic search
    :param atelier: an instance of the problem
    
    args
    :param max_gen: maximum number of generation
    :param max_time: maximum computation time
    
    :return: a dictionnary where the keys are the machines and the values are the sites of the machines"""

    # Initialisation

    population = generate_population(atelier, args)
    
    best_cost = 10**10
    best_solution = dict()

    for sol in population:
        cost = atelier.get_total_cost(sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = dict(sol)
    
    n_gen = 0
    t0 = t.time()

    while n_gen < args.max_gen and t.time()-t0 < args.max_time:
        
        # Selection
        population = roulette(atelier, population)

        # Hybridation
        population = crossing(atelier, population)

        # Mutation
        population = mutation(atelier, population, args)

        # Updating
        for sol in population:
            cost = atelier.get_total_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = dict(sol)
        
        n_gen += 1

    return best_solution