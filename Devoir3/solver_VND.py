from typing import List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np
from utils.utils import generate_random_valid_solution


def VND(r: RCPSP, time_limit=10*60):

    start_time = time()
    tic = start_time

    with tqdm(total=time_limit) as progress_bar:

        # Initial solution
        solution = generate_random_valid_solution(r)

        # Build the neighborhood priority list (priority to node that have the most sucessors)
        nodes = sorted(list(r.graph.nodes))
        del nodes[0]
        node_neighbor = sorted([(node_id, len(list(r.graph.successors(node_id)))) for node_id in nodes], key=lambda x: -x[1])
        print(node_neighbor)
        node_neighbor = [element[0] for element in node_neighbor]

        # Parameter
        k = 1
        k_max = len(node_neighbor) - 1

        while k != k_max:
        
            new_solution =  local_search(r, solution, k, node_neighbor)
            solution, k = neighborhood_change(r, solution, new_solution, k)
            
            if (tac:=time()) - start_time < time_limit:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                print("\nTime out - Returning current best\n")
                return solution

    return solution


def local_search(r: RCPSP, solution: list, k: int, node_neighbor: list):
    return solution


def neighborhood_change(r: RCPSP, solution: list, new_solution: list, k: int):
    if r.get_solution_cost(new_solution) < r.get_solution_cost(solution):
       return new_solution, k
    else:
        return solution, k+1


