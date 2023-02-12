"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""
from typing import List, Tuple, Set
from network import PCSTP
from tqdm import tqdm
import random
from copy import deepcopy
from utils.benchmarking import timer
import time


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
    ##? Starting with a arbitrary solution
    s = build_random_solution(pcstp)
    i = 0
    ##? Tant qu'il existe une solution dans le voisinage

    start = time.time()
    elapsed_time = 0
    max_time = 20 * 60

    while elapsed_time < max_time:
        ##? Changer la solution localement
        s_i = find_better_neighboor(s, pcstp)

        ##? Check if better
        s_i
        # print(
        #     f"{i} sol: {pcstp.get_solution_cost(s)} <= s_i {pcstp.get_solution_cost(s_i)}"
        # )

        if pcstp.get_solution_cost(s) <= pcstp.get_solution_cost(s_i):
            if random.random() > 0.4 and pcstp.verify_solution(s_i):
                break
            elif i > 1000:
                break
            else:
                if random.random() > 0.1 and pcstp.get_solution_cost(s_i) < 0:
                    s = s_i
                i += 1
                continue
        elif pcstp.get_solution_cost(s_i) > 0:
            s = s_i
            i = 0

        elapsed_time = time.time() - start

    print(elapsed_time)
    print(max_time)
    ##? Retourner s_i
    return list(s_i)


def build_full_solution(pcstp: PCSTP) -> List[Tuple[int]]:
    # Connect everything
    return [edge for edge in pcstp.network.edges]


def build_random_solution(pcstp: PCSTP) -> List[Tuple[int]]:
    # Connect everything
    return [edge for edge in pcstp.network.edges if random.choice([True, False])]


def find_better_neighboor(solution: List[Tuple[int]], pcstp: PCSTP) -> List[Tuple[int]]:
    if not len(solution):
        return
    ##? Choose a random node in the sol¨ution
    solution = deepcopy(solution)
    edge = random.choice(solution)
    node = random.choice(edge)

    temperature = 0.75

    ##? Local search -> Look at the neighboor and switch the connection with P(temperautre)
    for neighboor in pcstp.network.adj[node]:
        if random.random() < temperature:
            if (node, neighboor) in solution:
                solution.remove((node, neighboor))
            elif (neighboor, node) in solution:
                solution.remove((neighboor, node))
            else:
                ##? Check if a cicle is created
                new_connexion = (node, neighboor)
                solution.append(new_connexion)
                if pcstp.verify_solution(solution) or random.random() < 0.1:
                    continue
                else:
                    solution.remove(new_connexion)

    return solution
