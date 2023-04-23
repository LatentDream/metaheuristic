import numpy as np
import math
import random
import time

import local_search as ls

def simulated_annealing(magic_square, neighbour_function, validity_function, eval_function, t_init, alpha, max_iteration, seed):

    start_time = time.time()

    solution = ls.generate_random_solution(magic_square, seed)
    neighboorhood = neighbour_function(solution)
    valid_neigboorhood = validity_function(neighboorhood)

    i = 0
    temperature = t_init
    best_solution = solution

    for i in range(max_iteration):

        candidate = random.choices(valid_neigboorhood)[0]
        delta = eval_function(magic_square, candidate) - eval_function(magic_square, solution)
        prob = max(np.exp(-delta / temperature), 0.01) # Small improvement compared to the version of the slide: the probability of degradation cannot goes below 1%

        if delta < 0:
            solution = candidate
        
        elif np.random.binomial(1, prob):
            solution = candidate

        if eval_function(magic_square, solution) < eval_function(magic_square, best_solution):
            print("ITERATION", i)
            print("BEST SOLUTION UPDATED")
            print("number of conflicts on current best solution:", ls.eval_n_conflict(magic_square, best_solution))
            print("degree of conflicts on current best solution:", ls.eval_degree_conflict(magic_square, best_solution))
        
            best_solution = solution

        temperature = alpha * temperature
        neighboorhood = neighbour_function(solution)
        valid_neigboorhood = validity_function(neighboorhood)

    end_time = time.time() - start_time
        
        
    print("FINAL SOLUTION OBTAINED")
    print("number of restarts:", i)
    print("total execution time:", round(end_time, 2), "seconds")
    print("number of conflicts:", ls.eval_n_conflict(magic_square, best_solution))
    print("degree of conflicts:", ls.eval_degree_conflict(magic_square, best_solution))
    print(best_solution)
    print("-----")
