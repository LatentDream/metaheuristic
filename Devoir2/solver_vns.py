import random
import time
from typing import List
from tsptw import TSPTW
import random
from utils.ant import Ant
from copy import deepcopy
from math import inf

from utils.beam_search import ProbabilisticBeamSearch

l_rate = 0.1  # l_rate: the learning rate for pheromone values
tau_min = 0.001  # lower limit for the pheromone values
tau_max = 0.999  # upper limit for the pheromone values
determinism_rate = 0.2  # rate of determinism in the solution construction
beam_width = 1  # parameters for the beam procedure
mu = 4.0  # stochastic sampling parameter
max_children = 100  # stochastic sampling parameter
n_samples = 10  # stochastic sampling parameter
sample_percent = 100  # stochastic sampling parameter


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """

    global ant
    ant = Ant(tsptw, l_rate=l_rate, tau_max=tau_max, tau_min=tau_min)

    global pbs
    pbs = ProbabilisticBeamSearch(
        tsptw,
        ant,
        beam_width,
        determinism_rate,
        max_children,
        mu,
        n_samples,
        sample_percent,
    )

    return variable_neighborhood_search(tsptw)


def variable_neighborhood_search(tsptw: TSPTW):
    start_time = time.time()
    time_limit = 60 * 30

    time_constraints = tsptw.time_windows
    best_solution = greedy_tsp(tsptw, time_constraints)
    best_cost = tsptw.get_solution_cost(best_solution)
    print("Greedy Cost :", best_cost)
    print("Greedy Path", best_solution)

    # Initialize the candidate solution
    best_solution = candidate_solution
    best_cost = tsptw.get_solution_cost(best_solution)
    candidate_solution = best_solution

    # Set the initial neighborhood structure and size
    neighborhood_structure = 1
    neighborhood_size = 3

    # Perform the VNS algorithm
    iterations = 0
    no_progress_count = 0
    while time.time() - start_time < time_limit:
        # Generate a new candidate solution in the current neighborhood structure

        new_candidate_solution = generate_neighbor_solution(
            candidate_solution, neighborhood_structure, neighborhood_size
        )

        # Verify if the new candidate solution is valid
        if tsptw.verify_solution(new_candidate_solution):
            # Compute the cost of the new candidate solution
            new_candidate_cost = tsptw.get_solution_cost(new_candidate_solution)
            # Check if the new candidate solution is better than the current one
            if new_candidate_cost < tsptw.get_solution_cost(candidate_solution):
                candidate_solution = new_candidate_solution
                # Check if the new candidate solution is better than the best one found so far
                if new_candidate_cost < best_cost:
                    best_solution = new_candidate_solution
                    best_cost = new_candidate_cost
                    print(
                        "BEST SOLUTION FOUND : {} | COST {}".format(
                            best_solution, best_cost
                        )
                    )

                # Reset the neighborhood structure and size
                neighborhood_structure = 1
                neighborhood_size = 3

            else:
                # Increase the neighborhood structure
                neighborhood_structure += 1
                no_progress_count += 1

                if neighborhood_structure > 3:
                    # Reset the neighborhood structure and size
                    neighborhood_structure = 1
                    neighborhood_size = 3

        else:
            # Increase the neighborhood size
            neighborhood_size += 1
            no_progress_count += 1
            neighborhood_structure += 1
            if neighborhood_structure > 3:
                # Reset the neighborhood structure and size
                neighborhood_structure = 1
                neighborhood_size = 3

        iterations += 1

        # No progress made : restart on a new solution
        if no_progress_count == 3:
            no_progress_count = 0
            candidate_solution = generate_fit_solution()

    return best_solution


def generate_fit_solution():
    solution = pbs.beam_construct()
    return solution


def generate_neighbor_solution(solution, neighborhood_structure, neighborhood_size):
    if neighborhood_structure == 1:
        return swap_nodes(solution, neighborhood_size)
    elif neighborhood_structure == 2:
        return reverse_subsequence(solution, neighborhood_size)
    elif neighborhood_structure == 3:
        return relocate_subsequence(solution, neighborhood_size)
    else:
        raise ValueError("Invalid neighborhood structure")


def swap_nodes(solution, neighborhood_size):
    new_solution = solution.copy()
    n = len(solution)
    N = min(5, neighborhood_size)
    nodes_to_swap = random.sample(range(1, n - 1), N)
    nodes_to_swap.sort()
    for i in range(1, N // 2):
        a = nodes_to_swap[i]
        b = nodes_to_swap[N - i - 1]
        new_solution[a], new_solution[b] = new_solution[b], new_solution[a]
    return new_solution


def reverse_subsequence(solution, neighborhood_size):
    new_solution = solution[1:-1].copy()
    i = random.randint(0, len(solution) - neighborhood_size)
    new_solution[i : i + neighborhood_size] = reversed(
        new_solution[i : i + neighborhood_size]
    )
    new_solution.insert(0, 0)
    new_solution.append(0)
    return new_solution


def relocate_subsequence(solution, neighborhood_size):
    if len(solution) <= neighborhood_size + 2:
        return swap_nodes(solution, neighborhood_size)

    new_solution = solution[1:-1].copy()
    i = random.randint(0, len(solution) - neighborhood_size)
    j = random.randint(0, len(solution) - neighborhood_size)
    while abs(i - j) < min(len(solution), neighborhood_size):
        j = random.randint(0, len(solution) - neighborhood_size)
    subsequence = new_solution[i : i + neighborhood_size]
    del new_solution[i : i + neighborhood_size]
    new_solution[j:j] = subsequence
    new_solution.insert(0, 0)
    new_solution.append(0)
    return new_solution


def check_time_constraint(tsptw: TSPTW, node1, node2, timer, time_constraints):
    # Check if the time window of node2 is valid given that node1 was visited at the start of its time window
    travel_time = tsptw.graph[node1][node2]["weight"]
    arrival_time = max(timer + travel_time, time_constraints[node2][0])
    return arrival_time <= time_constraints[node2][1]


def greedy_tsp(tsptw: TSPTW, time_constraints):

    nodes = list(range(len(time_constraints)))
    nodes = sorted(nodes, key=lambda x: time_constraints[x][0])

    # Find the node with the earliest opening time
    first_node = nodes[0]

    current_time = 0
    solution = [first_node]

    for node in nodes:
        if node == first_node:
            continue

        if check_time_constraint(
            tsptw, solution[-1], node, current_time, time_constraints
        ):
            solution.append(node)
            current_time = max(
                current_time + tsptw.graph[solution[-2]][solution[-1]]["weight"],
                time_constraints[node][0],
            )
        else:
            continue
    solution.append(0)
    return solution
