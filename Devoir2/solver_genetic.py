from typing import List, Tuple
from tsptw import TSPTW
import time
import networkx as nx
import itertools
import random
import numpy as np
from math import inf, ceil, log2, exp, log
import solver_naive
import random


def generate_chromosome(tsptw: TSPTW):
    chromosome = list(range(1, tsptw.num_nodes))
    random.shuffle(chromosome)
    chromosome = [0] + chromosome + [0]
    return chromosome


def generate_random_valid_solution(tsptw: TSPTW) -> List[int]:
    """
    Generate a random valid solution to the TSPTW problem.
    """

    solution = [1]
    time_left = [0] * (tsptw.num_nodes + 1)
    time_left[1] = tsptw.time_windows[0][0]

    while len(solution) < tsptw.num_nodes:
        candidates = []
        for node in tsptw.graph.neighbors(solution[-1]):
            if node not in solution:
                candidate_time = max(
                    time_left[solution[-1]] + tsptw.graph[solution[-1]][node]["weight"],
                    tsptw.time_windows[node - 1][0],
                )
                if candidate_time <= tsptw.time_windows[node - 1][1]:
                    candidates.append(node)

        if candidates:
            next_node = random.choice(candidates)
            solution.append(next_node)
            time_left[next_node] = max(
                tsptw.time_windows[next_node - 1][0],
                time_left[solution[-2]]
                + tsptw.graph[solution[-2]][next_node]["weight"],
            )
        else:
            return generate_random_valid_solution(tsptw)

    return solution


def generate_population(tsptw: TSPTW, pop_size):
    "Generate a population made of unique random permutations, and then repair the solutions to make them satisfy the hard constraints"
    population = []
    while len(population) < pop_size:
        chromosome = generate_chromosome(tsptw)
        valid_chromosome = repair_search(tsptw, chromosome, max_iterations=1000)
        if valid_chromosome not in population:
            population.append(valid_chromosome)
        print("Valid chromosome found")
    return population


def fitness(tsptw: TSPTW, solution):
    return 1 / tsptw.get_solution_cost(solution)


def selection(tsptw: TSPTW, population, pop_size, tournament_size, tournament_accepted):
    # Returns a list of the selected unique best chromosomes :
    # The population size is preserved and the selection is made with a tournament
    selected = []

    if pop_size == len(population):
        return population

    # Tournament to select the best solutions among the new population
    while len(selected) < pop_size:
        subset = random.sample(diff(population, selected), tournament_size)
        fittest_solutions = sorted(
            subset, key=lambda s: fitness(tsptw, s), reverse=True
        )[:tournament_accepted]

        selected.extend(fittest_solutions)

    return selected


# Define the crossover function
def crossover(parent1, parent2):
    # Select a random crossover point
    crossover_point = random.randint(1, len(parent1) - 2)

    # Create the first child
    child1 = [0] + parent1[1 : crossover_point + 1]
    for gene in parent2[1:]:
        if gene not in child1:
            child1.append(gene)
        if len(child1) == len(parent1):
            break
    child1.append(0)

    # Create the second child
    child2 = [0] + parent2[1 : crossover_point + 1]
    for gene in parent1[1:]:
        if gene not in child2:
            child2.append(gene)
        if len(child2) == len(parent1):
            break
    child2.append(0)
    return child1, child2


def get_pmx_value(idx, parent1, parent2, child):
    """
    Helper function for PMX crossover to determine the value of a missing gene.

    Args:
        idx (int): The index of the missing gene.
        parent1 (list): The first parent chromosome.
        parent2 (list): The second parent chromosome.
        child (list): The child chromosome being constructed.

    Returns:
        The value of the missing gene based on the PMX.
    """
    value = parent1[idx]
    while value in child:
        idx = parent1.index(parent2[idx])
        value = parent1[idx]
    return value


def m_point_crossover(parent1, parent2, m):
    """
    Applies m-point crossover to two parents and returns two offspring.
    """
    size = len(parent1)
    crossover_points = sorted(random.sample(range(1, size), m))

    offspring1 = [-1] * size
    offspring2 = [-1] * size

    for i in range(m + 1):
        if i == 0:
            start = 0
        else:
            start = crossover_points[i - 1]

        if i == m:
            end = size
        else:
            end = crossover_points[i]

        offspring1[start:end] = parent1[start:end]
        offspring2[start:end] = parent2[start:end]

    # Fill in remaining positions with genes from the other parent
    for i in range(size):
        if offspring1[i] == -1:
            if parent2[i] not in offspring1:
                offspring1[i] = parent2[i]
            else:
                offspring1[i] = get_pmx_value(parent1[i], parent2, offspring1)

        if offspring2[i] == -1:
            if parent1[i] not in offspring2:
                offspring2[i] = parent1[i]
            else:
                offspring2[i] = get_pmx_value(parent2[i], parent1, offspring2)

    return offspring1, offspring2


# Define the mutation function
def mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        # Select two random points to swap
        idx1, idx2 = random.sample(range(1, len(chromosome) - 1), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


# Define the genetic algorithm function with restarts and a time limit
def genetic_algorithm(
    tsptw: TSPTW,
    num_generations,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
):
    best_solution = None
    best_cost = inf
    start_time = time.time()
    num_restarts = 0

    while time.time() - start_time < time_limit:
        # Generate the initial population
        print("CREATING NEW POPULATION")
        population = generate_population(tsptw, pop_size)
        print("POPULATION CREATED")
        # Iterate over the generations
        for i in range(num_generations):
            # Select the parents for the next generation
            # parents = selection(tsptw, population, pop_size // 2, tournament_size)
            parents = population
            # Create the offspring for the next generation
            offspring = []
            for j in range(len(parents) // 2):
                parent1 = parents[j]
                parent2 = parents[len(parents) - j - 1]
                child1, child2 = m_point_crossover(
                    parent1, parent2, min(8, pop_size // 2)
                )
                child1 = mutation(child1, mutation_rate)
                child1 = repair_search(tsptw, child1)
                child2 = mutation(child2, mutation_rate)
                child2 = repair_search(tsptw, child2)
                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = selection(
                tsptw,
                population + offspring,
                pop_size,
                tournament_size,
                tournament_accepted,
            )

            # Update the best solution found so far
            fitness_scores = [fitness(tsptw, chromosome) for chromosome in population]
            best_idx = np.argmax(fitness_scores)

            if tsptw.get_solution_cost(population[best_idx]) < best_cost:
                best_solution = population[best_idx]
                best_cost = tsptw.get_solution_cost(population[best_idx])
                num_restarts = 0
                print("BEST SOLUTION FOUND : COST {}".format(best_cost))
            else:
                num_restarts += 1
                if num_restarts % 10 == 0:
                    # print("NO IMPROVEMENT AFTER 10 GENERATIONS, RESTARTING...")
                    break

    return best_solution


def repair_search(
    tsptw: TSPTW, solution: List[int], max_iterations: int = 100
) -> List[int]:

    i = 0
    # Iterate over a fixed number of iterations
    while True:
        # Select a random pair of nodes to exchange in the solution
        node1 = random.randint(1, len(solution) - 2)  # exclude first and last nodes
        node2 = random.randint(1, len(solution) - 2)  # exclude first and last nodes
        while node1 == node2:
            node2 = random.randint(1, len(solution) - 2)

        # Swap the positions of the two nodes
        new_solution = solution.copy()
        new_solution[node1], new_solution[node2] = (
            new_solution[node2],
            new_solution[node1],
        )

        # If the new solution satisfies the constraints, add it to the valid solutions found
        if tsptw.verify_solution(new_solution):
            return new_solution

        # If no valid solution can be found with the solution : generate another
        if i % len(solution) ^ 2 == 0:
            solution = generate_chromosome(tsptw)

        i += 1

    # Return the valid solutions found
    return solutions


def diff(list1, list2):
    """Returns the difference between lists of lists a and b"""
    return [x for x in list1 if x not in list2]


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """

    mutation_rate = 0.01
    pop_size = tsptw.num_nodes
    tournament_size = ceil(pop_size / 4)
    tournament_accepted = ceil(tournament_size / 5)
    num_generations = 1000
    time_limit = 60

    return genetic_algorithm(
        tsptw,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=time_limit,
    )
