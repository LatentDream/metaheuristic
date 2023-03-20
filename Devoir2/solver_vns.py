import random
import time
from typing import List
from tsptw import TSPTW
import random


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    return variable_neighborhood_search(tsptw)


def variable_neighborhood_search(tsptw):
    start_time = time.time()
    time_limit = 60 * 5

    # Initialize the candidate solution
    candidate_solution = generate_random_solution(tsptw)
    best_solution = candidate_solution
    best_cost = tsptw.get_solution_cost(best_solution)

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
        # print(candidate_solution)

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
        if no_progress_count == len(best_solution):
            no_progress_count = 0
            candidate_solution = generate_random_solution(tsptw)

    return best_solution


def generate_random_solution(tsptw):
    solution = list(range(1, tsptw.num_nodes))
    random.shuffle(solution)
    solution = [0] + solution + [0]
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

    # print("sol", solution)
    # print("i", i)
    # print("j", j)
    # print("new", new_solution)
    return new_solution
