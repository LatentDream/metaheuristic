"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""
from typing import List, Tuple, Set
from network import PCSTP
from tqdm import tqdm
import random
from copy import deepcopy

def solve(pcstp: PCSTP) -> List[Tuple[int]]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph 
            are excluded
    """   
    ######! Local search heuristique
    ##? Starting with a arbitrary solution
    s = build_random_solution(pcstp)

    ##? Tant qu'il existe une solution dans le voisinage
    while True:
        ##? Changer la solution localement
        s_i = find_better_neighboor(s, pcstp)

        ##? Check if better
        print(f"sol: {pcstp.get_solution_cost(s)} <= s_i {pcstp.get_solution_cost(s_i)}")

        if pcstp.get_solution_cost(s) <=  pcstp.get_solution_cost(s_i):
            break
        elif pcstp.get_solution_cost(s_i) > 0:
            s = s_i

    ##? Retourner s_i
    return s_i


def build_random_solution(pcstp: PCSTP) -> Set[Tuple[int]]:
    # Connect everything
    return [edge for edge in pcstp.network.edges]


def find_better_neighboor(solution: Set[Tuple[int]], pcstp: PCSTP) -> Set[Tuple[int]]:
    if not len(solution): return
    ##? Choose a random node in the solution
    solution = deepcopy(solution)
    edge = random.choice(solution)
    node = random.choice(edge)

    temperature = 0.75

    ##? Local search -> Look at the neighboor and switch the connection with P(temperautre)
    for neighboor in pcstp.network.adj[node]:
        if (node, neighboor) in solution:
            if random.random() < temperature:
                solution.remove((node, neighboor))
        elif (neighboor, node) in solution:
            if random.random() < temperature:
                solution.remove((neighboor, node))

    return solution

