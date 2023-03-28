from typing import List, Tuple, Dict
from rcpsp import RCPSP
import time
import random
import numpy as np
import networkx as nx
import solver_naive
from copy import deepcopy
from math import inf, ceil


def solve(rcpsp: RCPSP) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        rcpsp (RCPSP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """

    time_limit = 5 * 60  # 20 * 60
    k_min = 1
    k_max = 10  # ceil(0.30 * len(rcpsp.graph.nodes))
    max_iter = 20

    return LNS_RCPSP(
        r=rcpsp, time_limit=time_limit, k_min=k_min, k_max=k_max, max_iter=max_iter
    )


def LNS_RCPSP(r: RCPSP, time_limit, k_min, k_max, max_iter):

    start_time = time.time()

    # Create an initial solution
    current_solution = generate_solution(r)
    best_solution = deepcopy(current_solution)
    best_fitness = fitness(r, best_solution)

    no_progress = 0
    while time.time() - start_time < time_limit:
        # Randomly choose a subproblem size between k_min and k_max
        k = random.randint(k_min, k_max)

        # Remove the selected tasks and their dependencies from the solution
        partial_solution = remove_tasks(current_solution, k)

        # Solve the subproblem with an exact method
        new_solution = local_search(r, partial_solution, max_iter=max_iter)
        new_fitness = fitness(r, new_solution)

        # Update the best solution if the new solution is better
        if new_fitness > best_fitness:
            print("best solution")
            best_solution = deepcopy(new_solution)
            best_fitness = new_fitness
            print(best_fitness)
            no_progress = 0

        no_progress += 1

        if no_progress % 10 == 0:
            current_solution = generate_solution(r)
        elif no_progress % 20 == 0:
            current_solution = best_solution
        else:
            current_solution = deepcopy(new_solution)

    return best_solution


###################################### Evaluation Functions ####################################


def fitness(r: RCPSP, solution):
    st_conflict = start_conflict(r, solution)
    prec_conflicts = precedence_conflicts(r, solution)
    res_conflicts = resource_conflicts(r, solution)

    fitness = -(
        r.get_solution_cost(solution)
        + 10 * st_conflict
        + 1000 * prec_conflicts
        + 100 * res_conflicts
    )
    return fitness


def start_conflict(r: RCPSP, solution):
    min_start_time = min([solution[job] for job in solution.keys()])
    if min_start_time != 0:
        return 1
    return 0


def precedence_conflicts(r: RCPSP, solution):
    n_conflicts = 0
    # Check precedence constraints
    for job in solution.keys():
        duration = r.graph.nodes[job]["duration"]
        job_start_time = solution[job]
        job_finish_time = job_start_time + duration
        for successor in r.graph.successors(job):
            if successor not in solution:
                n_conflicts += 1
                continue

            if solution[successor] < job_finish_time:
                n_conflicts += 1
    return n_conflicts


def resource_conflicts(r: RCPSP, solution):
    n_conflicts = 0
    # Check resource constraints
    num_resources = len(r.resource_availabilities)

    # Find the maximum finish time to set the range for resource usage check
    max_finish_time = max(
        [solution[job] + r.graph.nodes[job]["duration"] for job in solution.keys()]
    )

    for t in range(max_finish_time + 1):
        resource_usage = [0] * num_resources
        for job, start_time in solution.items():
            job_finish_time = start_time + r.graph.nodes[job]["duration"]
            if start_time <= t <= job_finish_time:  # Fix the condition here
                job_resources = nx.get_node_attributes(r.graph, "resources")[job]
                resource_usage = [x + y for x, y in zip(resource_usage, job_resources)]

        if any(
            usage > available
            for usage, available in zip(resource_usage, r.resource_availabilities)
        ):
            n_conflicts += 1
    return n_conflicts


def generate_solution(r):
    """Generate a random feasible solution for the RCPSP problem using earliest start time algorithm"""

    horizon = 181
    solution = {}

    # Sort tasks by their order in the graph, assuming the tasks are numbered sequentially
    tasks = list(r.graph.nodes)

    for task in tasks:
        solution[task] = random.choice([i for i in range(0, horizon + 1)])

    return solution


def remove_tasks(solution, n_remove):
    """
    Removes n_remove tasks from the solution

    Parameters:
    solution (dict): A dictionary representing the solution, mapping tasks to their start times
    n_remove (int): The number of tasks to remove from the solution

    Returns:
    dict: A new solution with n_remove tasks removed
    """

    # Get the list of tasks and their start times from the solution
    tasks = list(solution.keys())
    start_times = list(solution.values())

    # Choose n_remove tasks to remove
    remove_indices = random.sample(range(len(tasks)), n_remove)

    # Remove the chosen tasks and their start times from the lists
    for i in sorted(remove_indices, reverse=True):
        del tasks[i]
        del start_times[i]

    # Rebuild the solution dictionary
    new_solution = {}
    for i in range(len(tasks)):
        new_solution[tasks[i]] = start_times[i]

    return new_solution


def combine_solutions(solution1, solution2):
    """Combine two solutions by merging their mappings"""

    # Merge the two mappings into a new mapping
    new_mapping = {**solution1, **solution2}

    return new_mapping


def local_search(r: RCPSP, solution: Dict[int, int], max_iter=20):
    """
    Simple local search algorithm that adds missing tasks to the solution with the best fitness score
    """
    horizon = 181

    # Compute the list of missing tasks
    missing_tasks = [t for t in r.graph.nodes if t not in solution]
    # Iterate over missing tasks and add the one that improves the fitness score the most

    for task in missing_tasks:
        best_start_time = 0
        best_fitness = inf

        # Iterate over possible start times for the missing task
        for _ in range(max_iter):

            start_time = random.choice(range(0, horizon))
            # Compute the fitness score for the solution with the missing task added at this start time
            temp_solution = deepcopy(solution)
            temp_solution[task] = start_time

            fitness_score = fitness(r, temp_solution)

            # Update the best start time and fitness score if a better one is found
            if fitness_score > best_fitness:
                best_fitness = fitness_score
                best_start_time = start_time

        # Add the missing task to the solution with the best start time
        solution[task] = best_start_time

    return solution
