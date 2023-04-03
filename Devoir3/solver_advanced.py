from typing import List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np
from utils.utils import generate_random_valid_solution

def solve(rcpsp: RCPSP) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        rcpsp (RCPSP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation 
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    # Add here your solving process here
    return algo_name(rcpsp, time_limit=30)


def algo_name(r: RCPSP, time_limit=10*60):

    start_time = time()
    tic = start_time


    with tqdm(total=time_limit) as progress_bar:

        initial_solution = generate_random_valid_solution(r)

        if (tac:=time()) - start_time < time_limit:
            progress_bar.update(tac - tic)
            tic = tac
        else:
            print("\nTime out - Returning current best\n")
            return initial_solution

    print(initial_solution)
    return initial_solution

