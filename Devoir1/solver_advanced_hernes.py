"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""
from typing import List, Tuple
from network import PCSTP
import random
from copy import deepcopy
from utils.tree import build_valid_solution


def solve(pcstp: PCSTP, seed=0) -> List[Tuple[int]]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph 
            are excluded
    """
    if seed: 
        random.seed(seed)   

    ######! Local search heuristique
    ##? Starting with a arbitrary VALID solution
    root_node  = build_valid_solution(pcstp, True) 
    s = root_node.get_connection_list()
    print(s)
    return s

    ##? As long as there is a solution in the neighborhoods
    while True:

        ##? Change the solution locally
        s_i = find_better_local_solution(s, pcstp)

        ##? Check if better
        print(f"sol: {pcstp.get_solution_cost(s)} <= s_i {pcstp.get_solution_cost(s_i)}")
        if pcstp.get_solution_cost(s) <=  pcstp.get_solution_cost(s_i):
            break
        
    ##? Retourner s_i
    return list(s_i)
    


def find_better_local_solution(solution: List[Tuple[int]], pcstp: PCSTP) -> List[Tuple[int]]:
    if not len(solution): return
    ##? Choose a random node in the solution
    solution = deepcopy(solution) 
    edge = random.choice(solution)
    node = random.choice(edge)

    ##? Rebuild a representation of the tree with `node` as the root



    ##? Add or remove connection from the root to his children
    #?   -> Make sure node is not already in the tree (No cycle)


