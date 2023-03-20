from typing import List
from tsptw import TSPTW
import time
import random
import numpy as np
from math import inf, ceil
import random
from copy import deepcopy
from utils.ant import Ant
from copy import deepcopy
from utils.beam_search import ProbabilisticBeamSearch


def solve(tsptw: TSPTW) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """

    mutation_rate = 0.05
    pop_size = tsptw.num_nodes * 10
    tournament_size = ceil(pop_size / 10)
    tournament_accepted = ceil(tournament_size / 5)
    num_generations = 100
    time_limit = 60 * 5

    l_rate = 0.1  # l_rate: the learning rate for pheromone values
    tau_min = 0.001  # lower limit for the pheromone values
    tau_max = 0.999  # upper limit for the pheromone values
    determinism_rate = 0.2  # rate of determinism in the solution construction
    beam_width = 1  # parameters for the beam procedure
    mu = 4.0  # stochastic sampling parameter
    max_children = 100  # stochastic sampling parameter
    n_samples = 10  # stochastic sampling parameter
    sample_percent = 100  # stochastic sampling parameter

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

    return genetic_algorithm(
        tsptw,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=time_limit,
    )


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
        population = generate_population(tsptw, pop_size)
        # Iterate over the generations
        for _ in range(num_generations):

            # Select the parents for the next generation
            parents = sorted(population, key=lambda s: fitness(tsptw, s), reverse=True)[
                : pop_size // 3
            ]

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(parents) // 2):
                parent1 = parents[j]
                parent2 = parents[len(parents) - j - 1]
                child1, child2 = crossover(parent1, parent2)
                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
                offspring.append(child1)
                offspring.append(child2)

            population = parents + offspring

            while len(population) < pop_size:
                population.append(generate_fit_chromosome(tsptw))

            # Select the survivors for the next generation : we keep the same population size
            population = selection(
                tsptw,
                population,
                pop_size,
                tournament_size,
                tournament_accepted,
            )

            # Update the best solution found so far
            fitness_scores = [fitness(tsptw, chromosome) for chromosome in population]
            best_idx = np.argmax(fitness_scores)
            if (
                tsptw.verify_solution(population[best_idx])
                and tsptw.get_solution_cost(population[best_idx]) < best_cost
            ):
                best_solution = population[best_idx]
                best_cost = tsptw.get_solution_cost(population[best_idx])
                num_restarts = 0
                print("BEST SOLUTION FOUND : COST {}".format(best_cost))
            else:
                num_restarts += 1
                if num_restarts % 100 == 0:
                    # print("NO IMPROVEMENT AFTER 100 GENERATIONS, RESTARTING...")
                    break

    return best_solution


def generate_fit_chromosome(tsptw):
    return pbs.beam_construct()


def generate_population(tsptw, pop_size):
    population = []
    for _ in range(pop_size *5):
        population.append(generate_fit_chromosome(tsptw))

    population = sorted(
        population, key=lambda s: get_number_of_violations(s, tsptw), reverse=False
    )[:pop_size]

    return population


def fitness(tsptw: TSPTW, solution):
    return 1 / (get_number_of_violations(solution, tsptw) + 10e-10)


def get_number_of_violations(solution: List[int], tsptw: TSPTW) -> int:
    nb_of_violation = 0
    time_step = 0
    last_stop = 0
    for next_stop in solution[1:]:
        edge = (last_stop, next_stop)
        time_step += tsptw.graph.edges[edge]["weight"]
        time_windows_begening, time_windows_end = tsptw.time_windows[next_stop]
        if time_step < time_windows_begening:
            waiting_time = time_windows_begening - time_step
            time_step += waiting_time
        if time_step > time_windows_end:
            nb_of_violation += 1

    return nb_of_violation


def selection(tsptw: TSPTW, population, pop_size, tournament_size, tournament_accepted):
    # Returns a list of the selected unique best chromosomes :
    # The population size is preserved and the selection is made with a tournament
    selected = []

    # Tournament to select the best solutions among the new population
    while len(selected) < pop_size:

        subset = random.sample(population, tournament_size)
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


# Define the mutation function
def mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        # Select two random points to swap
        idx1, idx2 = random.sample(range(1, len(chromosome) - 1), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


def generate_chromosome(tsptw: TSPTW):
    chromosome = list(range(1, tsptw.num_nodes))
    random.shuffle(chromosome)
    chromosome = [0] + chromosome + [0]
    return chromosome
