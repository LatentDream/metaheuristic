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

    # Solve the border
    time_limit = 60  # 20 * 60

    pop_size = 1000
    mutation_rate = 0.5
    max_time_local_search = 1
    tournament_size = 100
    tournament_accepted = 20
    num_generations = 1000
    no_progress_generations = 100
    elite_size = 0

    border_solution, cost = genetic_algorithm_border(
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
    )
    visualize(e, border_solution, "BORDEEER")
    return border_solution, cost

    # Solve the inner puzzle

    time_limit = 60 * 5  # 20 * 60

    pop_size = 300
    mutation_rate = 0.5
    max_time_local_search = 1
    tournament_size = 20
    tournament_accepted = 5
    num_generations = 1000
    no_progress_generations = 5
    elite_size = 0

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
):

    start_time = time.time()
    best_fitness_no_improvement = -inf
    best_fitness = -inf
    improvement_timer = 1
    time_over = False

    while not time_over:

        # Generate the initial population
        population = generate_population(e, pop_size, elite_size=elite_size)
        population = sorted(population, key=lambda s: fitness(e, s), reverse=True)

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

                child1 = inner_crossover(
                    parent1,
                    parent2,
                    random.choice([i for i in range(2, 8)]),
                )
                child2 = inner_crossover(
                    parent1,
                    parent2,
                    random.choice([i for i in range(2, 8)]),
                )

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
            print(fittest_score)
            if fittest_score > best_fitness_no_improvement:

                improved_solution, improved_solution_score = local_search(
                    e, fittest_solution, max_time_local_search=max_time_local_search
                )
                if improved_solution_score > best_fitness:
                    best_fitness = improved_solution_score
                    best_solution = improved_solution.copy()
                    print(
                        "BEST SOLUTION FOUND : Cost ",
                        e.get_total_n_conflict(best_solution),
                    )

                    for instance, length in file_names.items():
                        if length == len(best_solution):
                            a = instance

                    visualize(
                        e,
                        best_solution,
                        "visualisation/{}/Cost {}".format(
                            a, e.get_total_n_conflict(best_solution)
                        ),
                    )

                improvement_timer = 0

            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer > no_progress_generations == 0:
                    improvement_timer = 0
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    return best_solution, e.get_total_n_conflict(best_solution)


###################################### Evaluation Functions ####################################


def fitness(e: EternityPuzzle, solution):
    return -e.get_total_n_conflict(solution)


###################################### Genetic Operaters ###############################


# Function to generate a random solution
def generate_random_solution(e: EternityPuzzle):

    solution = []
    remaining_piece = deepcopy(e.piece_list)

    for _ in range(e.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = e.generate_rotation(piece)[permutation_idx]
        solution.append(piece_permuted)
        remaining_piece.remove(piece)

    return solution


def generate_fit_chromosome(e: EternityPuzzle):
    solution, _ = solver_heuristic.solve_heuristic(e)
    return solution


# def find_best_position(e: EternityPuzzle, piece, solution):
#     best_position = None
#     best_conflict = float("inf")
#     for i in range(e.n_piece):
#         if solution[i] == (WHITE, WHITE, WHITE, WHITE):
#             conflicts = get_piece_conflicts(e, i, piece, solution)
#             if conflicts < best_conflict:
#                 best_position = i
#                 best_conflict = conflicts
#     return best_position


# def get_piece_conflicts(e: EternityPuzzle, position, piece, solution):
#     conflicts = 0
#     j, i = divmod(position, e.board_size)
#     if i > 0:
#         conflicts += int(piece[WEST] != solution[position - 1][EAST])
#     if j > 0:
#         conflicts += int(piece[NORTH] != solution[position - e.board_size][SOUTH])
#     return conflicts


def generate_population(e: EternityPuzzle, pop_size, elite_size):
    population = []

    for _ in range(elite_size):
        population.append(generate_fit_chromosome(e))

    for _ in range(pop_size):
        population.append(generate_random_solution(e))
    return population


def inner_crossover(parent1, parent2, num_points):
    # Choose random crossover points
    length = len(parent1)
    points = sorted(random.sample(range(length), num_points))

    print(parent1)
    pieces = set(parent1)

    # Create child using parent1's genes in selected segments
    child = parent1.copy()
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        child[start:end] = parent2[start:end]

    return child


def mutation(e, solution, mutation_rate):
    # Rotate a piece

    mutated_solution = deepcopy(solution)

    for idx, piece in enumerate(mutated_solution):
        # Rotate piece
        if random.random() < mutation_rate:
            mutated_solution[idx] = random.choice(e.generate_rotation(piece))

        # # 2-swap pieces :
        # if random.random() < mutation_rate:
        #     # choose two distinct indices to swap
        #     index1 = random.randint(0, len(solution) - 1)
        #     index2 = random.randint(0, len(solution) - 1)
        #     # swap the tiles at the chosen indices
        #     mutated_solution[index1], mutated_solution[index2] = (
        #         mutated_solution[index2],
        #         mutated_solution[index1],
        #     )
    return mutated_solution


# Selection through a tournament
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


# def encode_solution(e: EternityPuzzle, solution):
#     """Encodes a solution as a list of chromosomes, where each chromosome represents a piece on the board."""
#     chromosomes = []
#     for y in range(e.board_size):
#         for x in range(e.board_size):
#             piece = []
#             for j in range(4):
#                 for i in range(4):
#                     piece.append(solution)
#                     piece.append(color)
#             chromosomes.append(piece)
#     return chromosomes


# def decode_solution(e: EternityPuzzle, chromosome):
#     """Decodes a list of chromosomes into a 2D array representing the solved board."""
#     board = [[-1 for x in range(e.board_size)] for y in range(e.board_size)]
#     chromosome_index = 0
#     for y in range(e.board_size):
#         for x in range(e.board_size):
#             # Only decode pieces that are not already on the board
#             if board[y][x] == -1:
#                 piece = chromosome[chromosome_index]
#                 for j in range(4):
#                     for i in range(4):
#                         color = piece[j * 4 + i]
#                         board[y + j][x + i] = color
#                 chromosome_index += 1
#     return board


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
        population = generate_border_population(e, pop_size)
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

                child1 = mutation(e, child1, mutation_rate)
                child2 = mutation(e, child2, mutation_rate)

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
            print(fittest_score)
            if fittest_score > best_fitness_no_improvement:

                # improved_solution, improved_solution_score = local_search(
                #     e, fittest_solution, max_time_local_search=max_time_local_search
                # )
                # todo : local search for border
                improved_solution, improved_solution_score = (
                    fittest_solution,
                    fittest_score,
                )

                if improved_solution_score > best_fitness:
                    best_fitness = improved_solution_score
                    best_solution = improved_solution.copy()

                improvement_timer = 0

            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer > no_progress_generations == 0:
                    improvement_timer = 0
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    return best_solution, fitness_border(e, best_solution)


def border_crossover(e: EternityPuzzle, parent):
    b = e.board_size
    l = len(parent)

    # Select randomly two locations on the border
    i = random.choice(range(b))
    j = random.choice(range(b))
    while not (i % b == 0 or l - i <= b or i <= b - 1 or i % b - 1 == 0):
        i = random.choice(range(b))
    while not (j % b == 0 or l - j <= b or j <= b - 1 or j % b - 1 == 0):
        j = random.choice(range(b))

    # 2-swap
    child = parent.copy()
    child[i], child[j] = parent[j], parent[i]

    return child


def fitness_border(e: EternityPuzzle, s):
    border_copy = deepcopy(s)
    b = e.board_size

    # Set all the inner pieces to black to ignore them in the cost
    for i in range(e.n_piece):
        if not (i % b == 0 or i < b - 1 or len(s) - 1 <= b or i % (b - 1) == 0):
            border_copy[i] = (BLACK, BLACK, BLACK, BLACK)

    border_cost = e.get_total_n_conflict(border_copy) - 4 * (e.board_size - 2)
    return -border_cost


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


def generate_border_population(e: EternityPuzzle, pop_size):
    population = []

    for _ in range(pop_size):
        population.append(generate_random_border(e))
    return population


def generate_random_border(e: EternityPuzzle):

    solution = generate_random_solution(e)

    gray_inner_pieces = []
    no_gray_border_pieces = []

    b = e.board_size

    # All the pieces with gray color must be on the border
    for i in range(e.n_piece):
        if not (i % b == 0 or i <= b - 1 or i + b >= e.n_piece or (i + 1) % b == 0):
            if GRAY in solution[i]:
                gray_inner_pieces.append(i)
        else:
            if not (GRAY in solution[i]):
                no_gray_border_pieces.append(i)

    random.shuffle(gray_inner_pieces)
    random.shuffle(no_gray_border_pieces)

    while len(no_gray_border_pieces) > 0:
        i_c = no_gray_border_pieces.pop()
        i_g = gray_inner_pieces.pop()
        solution[i_c], solution[i_g] = solution[i_g], solution[i_c]

    return solution
