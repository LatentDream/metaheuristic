from typing import List, Tuple
from tsptw import TSPTW
import time
import networkx as nx
import itertools
import random
import numpy as np
from math import inf, ceil, log2, exp, log


def generate_population(tsptw: TSPTW, pop_size):
    "Generate the population randomly, without satisfying the time constraints"
    population = []
    for _ in range(pop_size):
        chromosome = list(range(1, tsptw.num_nodes))
        random.shuffle(chromosome)
        chromosome = [0] + chromosome + [0]
        population.append(chromosome)
    return population


def fitness(tsptw: TSPTW, solution):
    # Compute the distance traveled by the salesman
    if not tsptw.verify_solution(solution):
        return 0
    total_distance = tsptw.get_solution_cost(solution)
    return 1 / total_distance


def selection(tsptw: TSPTW, population, tournament_size):
    selected = []
    for _ in range(len(population)):
        subset = random.sample(population, tournament_size)
        fitness_scores = [fitness(tsptw, chromosome) for chromosome in subset]
        fittest_idx = np.argmax(fitness_scores)
        selected.append(subset[fittest_idx])
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


# Define the mutation function
def mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        # Select two random points to swap
        idx1, idx2 = random.sample(range(1, len(chromosome) - 1), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


# Define the genetic algorithm function with restarts and a time limit
def genetic_algorithm(
    tsptw: TSPTW, num_generations, mutation_rate, tournament_size, pop_size, time_limit
):
    best_solution = None
    best_cost = inf
    start_time = time.time()
    num_restarts = 0

    while time.time() - start_time < time_limit:
        # Generate the initial population
        population = generate_population(tsptw, pop_size)
        # Iterate over the generations
        for i in range(num_generations):
            # Select the parents for the next generation
            parents = selection(tsptw, population, tournament_size)
            # Create the offspring for the next generation
            offspring = []
            for j in range(len(parents) // 2):
                parent1 = parents[j]
                parent2 = parents[len(parents) - j - 1]
                child1, child2 = crossover(parent1, parent2)
                offspring.append(mutation(child1, mutation_rate))
                offspring.append(mutation(child2, mutation_rate))

            # Select the survivors for the next generation : we keep the same population size
            population = selection(
                tsptw,
                population[: len(population) - len(offspring)] + offspring,
                pop_size,
            )

            # Update the best solution found so far
            fitness_scores = [
                fitness(tsptw, chromosome)
                for chromosome in population
                if tsptw.verify_solution(chromosome)
            ]
            if len(fitness_scores) > 0:
                best_idx = np.argmax(fitness_scores)
                if tsptw.get_solution_cost(population[best_idx]) < best_cost:
                    best_solution = population[best_idx]
                    best_cost = tsptw.get_solution_cost(population[best_idx])
                    num_restarts = 0
                    print("BEST SOLUTION FOUND : COST {}".format(best_cost))
                else:
                    num_restarts += 1
                    if num_restarts % 10 == 0:
                        print("NO IMPROVEMENT AFTER 10 GENERATIONS, RESTARTING...")
                        break
            else:
                num_restarts += 1
                if num_restarts % 10 == 0:
                    print("NO IMPROVEMENT AFTER 10 GENERATIONS, RESTARTING...")
                    break

    return best_solution


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
    pop_size = 100  # ceil(log(tsptw.num_nodes))
    tournament_size = pop_size // 2
    num_generations = 10
    time_limit = 60

    return genetic_algorithm(
        tsptw,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        pop_size=pop_size,
        time_limit=time_limit,
    )
