import numpy as np
import math
import random
import time

import local_search as ls

def local_search_with_restart(magic_square, neighbour_function, validity_function, eval_function, seed, n_restart):

    start_time = time.time()
    best_solution = ls.generate_random_solution(magic_square, seed)

    for i in range(n_restart):

        # Note that the seed for the generation of the random instance changes at each iteration
        solution = ls.local_search(magic_square, neighbour_function, validity_function, eval_function, i)

        if eval_function(magic_square, solution) < eval_function(magic_square, best_solution):
            print("BEST SOLUTION UPDATED")
            best_solution = solution

        if eval_function(magic_square, best_solution) == 0:
            print("FEASIBLE SOLUTION FOUND")
            break

        print("RESTART: ", i)
    end_time = time.time() - start_time
        
        
    print("FINAL SOLUTION OBTAINED")
    print("number of restarts:", i)
    print("total execution time:", round(end_time, 2), "seconds")
    print("number of conflicts:", ls.eval_n_conflict(magic_square, best_solution))
    print("degree of conflicts:", ls.eval_degree_conflict(magic_square, best_solution))
    print(best_solution)
    print("-----")
