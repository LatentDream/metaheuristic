from copy import copy
from typing import Dict
from rcpsp import RCPSP
from time import time
from tqdm import tqdm
import random
import numpy as np
from utils.tree_node import TreeNode
from utils.utils import generate_random_valid_solution
from utils.vnd_utils import local_search


def solve(rcpsp: RCPSP):

    time_limit =  20 * 60 # In seconde

    # return VND(rcpsp, time_limit, build_neighbor_priority_with_stochasticity)
    return VND(rcpsp, time_limit, build_neighbor_priority)


def VND(r: RCPSP, time_limit, build_neighbor):

    start_time = time()
    tic = start_time

    with tqdm(total=time_limit) as progress_bar:

        best_solution = generate_random_valid_solution(r)

        while True:

            # Initial solution
            solution = generate_random_valid_solution(r)
            node_neighbor = build_neighbor(r)

            # Parameter : Neighborhood structure
            k = 0
            k_max = len(node_neighbor) - 1

            while k != k_max:
                new_solution = local_search(r, solution, k, node_neighbor)
                solution, k = neighborhood_change(r, solution, new_solution, k)
                
            if r.verify_solution(solution): 
                if r.get_solution_cost(solution) < r.get_solution_cost(best_solution):
                    best_solution = copy(solution)
            else:
                print("/!\ Warning: invalid solution detected /!\ ")

            if (tac := time()) - start_time < time_limit:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                break
            # break

    return best_solution


def build_random_neighbor_priority(r: RCPSP):
    nodes = [i + 1 for i in range(1, r.graph.number_of_nodes())]
    priority_neighbord = []
    for _ in range(len(nodes)):
        nodes_idx = random.randrange(0, len(nodes))
        priority_neighbord.append(nodes[nodes_idx])
        del nodes[nodes_idx]
    return priority_neighbord


def build_neighbor_priority_with_stochasticity(r: RCPSP):
    # Build the neighborhood priority list (priority to node that have the most sucessors)
    nodes = [i + 1 for i in range(1, r.graph.number_of_nodes())]
    node_neighbor = sorted(
        [
            (
                node_id,
                len(list(r.graph.successors(node_id))) + np.random.uniform(0.0, 0.5),
            )
            for node_id in nodes
        ],
        key=lambda x: -x[1],
    )
    node_neighbor = [element[0] for element in node_neighbor]
    return node_neighbor


def build_neighbor_priority(r: RCPSP):
    # Build a acyclic graph representaiton of the job dependencies
    nodes = {i: TreeNode(i) for i in range(1, r.graph.number_of_nodes() + 1)}
    for i in range(1, r.graph.number_of_nodes() + 1):
        children_id = r.graph.successors(i)
        nodes[i].add_children(*[nodes[child_id] for child_id in children_id])
    # fin number of job waiting for the job_i to be done
    node_priority = [
        (i, nodes[i].depth()) for i in range(1, r.graph.number_of_nodes() + 1)
    ]
    node_neighbor = sorted(node_priority, key=lambda x: x[1])
    return [element[0] for element in node_neighbor]


def neighborhood_change(
    r: RCPSP, solution: Dict[int, int], new_solution: Dict[int, int], k: int
):
    if r.get_solution_cost(new_solution) < r.get_solution_cost(solution):
        return new_solution, k
    else:
        return solution, k + 1


