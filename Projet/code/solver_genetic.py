import numpy as np
from copy import deepcopy
import time
import random
from utils.utils import *
from math import inf
from typing import Dict
import time
import random
import numpy as np
import networkx as nx
from math import inf
import solver_heuristic
import solver_local_search
import os
import json


file_names = {"A": 16, "B": 49, "C": 64, "D": 81, "E": 100, "complet": 256}

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


def solve_advanced(e: EternityPuzzle):
    """
    Your solver for the problem
    :param e: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    if os.path.exists("heuristic_solution.json"):
        os.remove("heuristic_solution.json")

    random.seed(1234)

    # Solve the border
    border_time = 1
    pop_size = 20
    mutation_rate = 0
    max_time_local_search = 30
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 10
    elite_size = 1

    border, border_cost = genetic_algorithm_border(
        e,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        max_time_local_search=max_time_local_search,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=border_time,
    )
    print("Border final cost : {}".format(border_cost))
    visualize(e, border, "Border")

    # Solve the inner puzzle
    time_limit = 20 * 60  # 20 * 60

    pop_size = 100
    mutation_rate = 1
    max_time_local_search = 20
    tournament_size = 100
    tournament_accepted = 30
    num_generations = 1000
    no_progress_generations = 1000
    elite_size = 1

    return genetic_algorithm(
        e,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        max_time_local_search=max_time_local_search,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=time_limit,
        border=border,
    )


def genetic_algorithm(
    e: EternityPuzzle,
    num_generations,
    no_progress_generations,
    max_time_local_search,
    elite_size,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
    border,
):
    start_time = time.time()
    best_fitness_no_improvement = -inf
    best_fitness = -inf
    improvement_timer = 0
    time_over = False

    while not time_over:
        # Generate the initial population
        population = generate_population(
            e, pop_size, elite_size=elite_size, border=border
        )
        population = sorted(population, key=lambda s: fitness(e, s), reverse=True)

        # Iterate over the generations
        for _ in range(num_generations):
            # The parents selected for the next generation
            parents = population[: pop_size // 2]

            # The elite is kept intact for the next generation
            elite = population[:elite_size]

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(parents) // 2):
                parent1 = parents[j]
                parent2 = parents[len(parents) - j - 1]

                child1 = inner_crossover(e, parent1)
                child2 = inner_crossover(e, parent2)

                child1 = mutation(e, child1, mutation_rate)
                child2 = mutation(e, child2, mutation_rate)

                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = (
                selection(
                    e,
                    parents + offspring,
                    pop_size,
                    tournament_size,
                    tournament_accepted,
                )
                + elite
            )

            population = sorted(population, key=lambda s: fitness(e, s), reverse=True)

            # Update the best solution found so far
            fittest_solution = population[0]
            fittest_score = fitness(e, fittest_solution)
            # print(fittest_score)
            if fittest_score > best_fitness_no_improvement:
                best_fitness_no_improvement = fittest_score
                improved_solution, improved_fitness = local_search(
                    e, fittest_solution, max_time_local_search=max_time_local_search
                )
                population.insert(0, improved_solution)

                if improved_fitness > best_fitness:
                    best_fitness = improved_fitness
                    best_solution = deepcopy(improved_solution)
                    best_cost = e.get_total_n_conflict(best_solution)
                    print("BEST SOLUTION FOUND : Cost ", best_cost)

                    for instance, length in file_names.items():
                        if length == len(best_solution):
                            instance_name = instance
                    visualize(
                        e,
                        best_solution,
                        "visualisation/{}/Cost {}".format(instance_name, best_cost),
                    )

                improvement_timer = 0

            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer > no_progress_generations:
                    improvement_timer = 0
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    return best_solution, e.get_total_n_conflict(best_solution)


###################################### Evaluation Functions ####################################


def fitness(e: EternityPuzzle, solution):
    return -e.get_total_n_conflict(solution)


def fitness_border(e: EternityPuzzle, s):
    border_copy = deepcopy(s)
    b = e.board_size

    # Set all the inner pieces to black to ignore them in the cost
    for i in range(e.n_piece):
        if not (i % b == 0 or i <= b - 1 or e.n_piece <= b + i or (i + 1) % b == 0):
            border_copy[i] = (BLACK, BLACK, BLACK, BLACK)

    border_cost = e.get_total_n_conflict(border_copy) - 4 * (e.board_size - 2)
    return -border_cost


###################################### Genetic Operaters ###############################


def generate_population(e: EternityPuzzle, pop_size, elite_size, border=None):
    population = []

    if border != None:
        for _ in range(elite_size):
            population.append(border)
        for _ in range(pop_size):
            population.append(generate_random_inner_solution(e, border))

    else:
        for _ in range(elite_size):
            solution_heuristic = get_heuristic_solution(e)
            population.append(solution_heuristic)
        for _ in range(pop_size):
            population.append(generate_random_solution(e))

    return population


# 2-Swap with rotations on conflict pieces
def inner_crossover(e: EternityPuzzle, parent):
    # 2-Swap with random rotations
    child = parent.copy()

    inner_idx = [
        i
        for i in range(e.board_size, e.n_piece - e.board_size)
        if not (i % e.board_size == 0 or (i + 1) % e.board_size == 0)
    ]

    conflict_positions = get_conflict_positions(e, parent)

    # 2- Swap conflict with a conflict piece
    for i in conflict_positions:
        if i in inner_idx:
            j = random.choice(inner_idx)
            child[i], child[j] = random.choice(
                e.generate_rotation(parent[j])
            ), random.choice(e.generate_rotation(parent[i]))

    # 2-Swap with random pieces
    for i in inner_idx:
        j = random.choice(inner_idx)
        child[i], child[j] = random.choice(
            e.generate_rotation(parent[j])
        ), random.choice(e.generate_rotation(parent[i]))

    return child


# 2-swap edges and orient the GRAY side for both pieces
def border_crossover(e: EternityPuzzle, parent):
    child = deepcopy(parent)

    edge_idx = [
        i
        for i in range(1, e.n_piece - 1)
        if i < e.board_size - 1
        or i % e.board_size == 0
        or e.n_piece - i < e.board_size
        or (i + 1) % e.board_size == 0
    ]

    conflict_positions = get_conflict_positions(e, parent)

    # 2- Swap conflict with a conflict piece
    for i in conflict_positions:
        if i in edge_idx:
            j = random.choice(edge_idx)
            child[i], child[j] = swap_orientations(e, parent[j], parent[i])

    # 2-Swap with random pieces
    for i in edge_idx:
        j = random.choice(edge_idx)
        child[i], child[j] = swap_orientations(e, parent[j], parent[i])

    return child


def mutation(e: EternityPuzzle, solution, mutation_rate):
    # Rotation of pieces
    mutated_solution = deepcopy(solution)
    for idx, piece in enumerate(mutated_solution):
        if random.random() < mutation_rate and piece_type(piece) == "inner":
            mutated_solution[idx] = random.choice(e.generate_rotation(piece)[1:])

    return mutated_solution


# Selection for inner pieces
def selection(
    e: EternityPuzzle, population, pop_size, tournament_size, tournament_accepted
):
    selected = []
    while len(selected) < pop_size:
        subset = random.sample(population, tournament_size)
        selected.extend(
            sorted(subset, key=lambda s: fitness(e, s), reverse=True)[
                :tournament_accepted
            ]
        )
    return selected


# Selection for border pieces
def selection_border(
    e: EternityPuzzle, population, pop_size, tournament_size, tournament_accepted
):
    selected = []
    while len(selected) < pop_size:
        subset = random.sample(population, tournament_size)
        selected.extend(
            sorted(subset, key=lambda s: fitness_border(e, s), reverse=True)[
                :tournament_accepted
            ]
        )
    return selected


################################# Local Search to improve valid solutions ##############################################
def local_search(
    e: EternityPuzzle, solution: Dict[int, int], max_time_local_search
) -> Dict[int, int]:
    best_solution, cost = solver_local_search.local_search(
        e, solution, max_time_local_search
    )
    return best_solution, -cost


#################################### Genetic algorithmm for the border ##################################################


def genetic_algorithm_border(
    e: EternityPuzzle,
    num_generations,
    no_progress_generations,
    max_time_local_search,
    elite_size,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
):
    start_time = time.time()
    best_fitness_no_improvement = -inf
    best_fitness = -inf
    improvement_timer = 1
    time_over = False

    while not time_over:
        # Generate the initial population
        population = generate_population(e, pop_size, elite_size)
        population = sorted(
            population, key=lambda s: fitness_border(e, s), reverse=True
        )

        # Iterate over the generations
        for _ in range(num_generations):
            # The parents selected for the next generation
            parents = population[: pop_size // 2]

            # The elite is kept for the next generation
            elite = population[:elite_size]

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(parents) // 2):
                parent1 = parents[j]
                parent2 = parents[len(parents) - j - 1]

                child1 = border_crossover(
                    e,
                    parent1,
                )
                child2 = border_crossover(
                    e,
                    parent2,
                )

                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = (
                selection_border(
                    e,
                    parents + offspring,
                    pop_size,
                    tournament_size,
                    tournament_accepted,
                )
                + elite
            )

            population = sorted(
                population, key=lambda s: fitness_border(e, s), reverse=True
            )

            # Update the best solution found so far
            fittest_solution = population[0]
            fittest_score = fitness_border(e, fittest_solution)
            # print(fittest_score)
            if fittest_score > best_fitness_no_improvement:
                # todo : local search for border
                # improved_solution, improved_solution_score = local_search(
                #     e, fittest_solution, max_time_local_search=max_time_local_search
                # )
                improved_solution, improved_solution_score = (
                    fittest_solution,
                    fittest_score,
                )
                population.insert(0, improved_solution)
                if improved_solution_score > best_fitness:
                    best_fitness = improved_solution_score
                    best_solution = improved_solution.copy()
                    print("New border cost : ", -best_fitness)
                improvement_timer = 0

            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer > no_progress_generations:
                    improvement_timer = 0
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    return best_solution, e.get_total_n_conflict(best_solution)


def get_heuristic_solution(e: EternityPuzzle):
    if os.path.exists("heuristic_solution.json"):
        with open("heuristic_solution.json", "r") as f:
            solution = json.load(f)
            solution = [tuple(sublst) for sublst in solution]

    else:
        solution = solver_heuristic.solve_heuristic(e)[0]
        with open("heuristic_solution.json", "w") as f:
            json.dump(solution, f)

    return solution
