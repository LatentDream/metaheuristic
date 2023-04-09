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
    random.seed(1234)

    # Solve the border
    border_time = 60
    pop_size = 500
    mutation_rate = 0
    max_time_local_search = 1
    tournament_size = 100
    tournament_accepted = 20
    num_generations = 100
    no_progress_generations = 5
    elite_size = 5

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
    print("Border cost : {}".format(border_cost))
    visualize(e, border, "Border")

    # Solve the inner puzzle
    time_limit = 20 * 60  # 20 * 60

    pop_size = 1000
    mutation_rate = 0.5
    max_time_local_search = 5
    tournament_size = 100
    tournament_accepted = 20
    num_generations = 1000
    no_progress_generations = 100
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
    improvement_timer = 1
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

            # The elite is kept for the next generation
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
                improved_solution, improved_solution_score = local_search(
                    e, fittest_solution, max_time_local_search=max_time_local_search
                )

                improved_solution, improved_solution_score = (
                    fittest_solution,
                    fittest_score,
                )

                population.insert(0, improved_solution)
                if improved_solution_score > best_fitness:
                    best_fitness = improved_solution_score
                    best_solution = improved_solution.copy()
                    print(
                        "BEST SOLUTION FOUND : Cost ",
                        e.get_total_n_conflict(best_solution),
                    )

                    for instance, length in file_names.items():
                        if length == len(best_solution):
                            instance_name = instance
                    visualize(
                        e,
                        best_solution,
                        "visualisation/{}/Cost {}".format(
                            instance_name, e.get_total_n_conflict(best_solution)
                        ),
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


def generate_random_solution(e: EternityPuzzle):
    """Constraints :
    Corners are on the corners and are well oriented
    Edges are on the edges and are well oriented
    """
    solution = [(BLACK, BLACK, BLACK, BLACK) for _ in range(e.n_piece)]

    remaining_piece = e.piece_list

    corners = [piece for piece in remaining_piece if piece_type(piece) == "corner"]
    random.shuffle(corners)

    edges = [piece for piece in remaining_piece if piece_type(piece) == "edge"]
    random.shuffle(edges)

    inner = [
        piece
        for piece in remaining_piece
        if ((piece not in corners) and (piece not in edges))
    ]
    random.shuffle(inner)

    b = e.board_size

    for i in range(e.n_piece):
        # Bottom-Left corner
        if i == 0:
            piece = corners.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [
                p for p in orientations if p[SOUTH] == GRAY and p[WEST] == GRAY
            ][0]
            solution[i] = oriented_piece

        # Bottom-Right corner
        elif i == b - 1:
            piece = corners.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [
                p for p in orientations if p[SOUTH] == GRAY and p[EAST] == GRAY
            ][0]
            solution[i] = oriented_piece

        # Top-Left corner
        elif i == e.n_piece - b:
            piece = corners.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [
                p for p in orientations if p[NORTH] == GRAY and p[WEST] == GRAY
            ][0]
            solution[i] = oriented_piece

        # Top-Right corner
        elif i == e.n_piece - 1:
            piece = corners.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [
                p for p in orientations if p[NORTH] == GRAY and p[EAST] == GRAY
            ][0]
            solution[i] = oriented_piece

        # Bottom edges :
        elif i < b - 1 and i != 0:
            piece = edges.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [p for p in orientations if p[SOUTH] == GRAY][0]
            solution[i] = oriented_piece

        # Left edges :
        elif i % b == 0 and i != 0 and i != e.n_piece - b:
            piece = edges.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [p for p in orientations if p[WEST] == GRAY][0]
            solution[i] = oriented_piece

        # Right edges :
        elif (i + 1) % b == 0 and i != b - 1 and i != e.n_piece - 1:
            piece = edges.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [p for p in orientations if p[EAST] == GRAY][0]
            solution[i] = oriented_piece

        # Top edges :
        elif i + b > e.n_piece and i != e.n_piece - 1:
            piece = edges.pop()
            orientations = e.generate_rotation(piece)
            oriented_piece = [p for p in orientations if p[NORTH] == GRAY][0]
            solution[i] = oriented_piece

        else:
            piece = inner.pop()
            solution[i] = e.generate_rotation(piece)[np.random.choice(np.arange(4))]

    return solution


def generate_random_inner_solution(e: EternityPuzzle, border):
    """Generate a random solution but keeps the border intact"""

    solution = deepcopy(border)
    b = e.board_size

    # IDs of the inner pieces
    inner_ids = [
        i
        for i in range(e.n_piece)
        if not (i <= b - 1 or i % b == 0 or e.n_piece - i <= b or (i + 1) % b == 0)
    ]
    inner_pieces = [solution[i] for i in inner_ids]
    random.shuffle(inner_pieces)

    for i, position in enumerate(inner_ids):
        solution[position] = inner_pieces[i]

    return solution


def generate_population(e: EternityPuzzle, pop_size, elite_size, border=None):
    population = []

    if border:
        for _ in range(elite_size):
            solution_heuristic, _ = solver_heuristic.solve_heuristic(e)
            population.append(solution_heuristic)

        for _ in range(pop_size):
            population.append(generate_random_inner_solution(e, border))

    else:
        for _ in range(elite_size):
            solution_heuristic, _ = solver_heuristic.solve_heuristic(e)
            population.append(solution_heuristic)

        for _ in range(pop_size):
            population.append(generate_random_solution(e))

    return population


def inner_crossover(e: EternityPuzzle, parent):
    # 2-Swap with random rotations
    b = e.board_size
    child = parent.copy()

    valid_ids = [
        i
        for i in range(e.n_piece)
        if not (i <= b - 1 or i % b == 0 or e.n_piece - i <= b or (i + 1) % b == 0)
    ]

    conflict_positions = get_conflict_positions(e, parent)

    # 2- Swap conflict with a conflict piece
    for _ in range(len(conflict_positions)):
        i, j = random.sample(valid_ids, 2)

        if i in conflict_positions or j in conflict_positions:
            child[i], child[j] = random.choice(
                e.generate_rotation(parent[j])
            ), random.choice(e.generate_rotation(parent[i]))

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
                    print("Border improved : ", best_fitness)
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

    return best_solution, fitness_border(e, best_solution)


def border_crossover(e: EternityPuzzle, parent):
    b = e.board_size
    child = parent.copy()

    # 2-swap between 2 randomly selected locations on the edges and not in the corner
    valid_ids = [
        i
        for i in range(1, e.n_piece - 1)
        if i < b - 1 or i % b == 0 or e.n_piece - i < b or (i + 1) % b == 0
    ]

    conflict_positions = get_conflict_positions(e, parent)

    for _ in range(len(conflict_positions)):
        i, j = random.sample(valid_ids, 2)

        if i in conflict_positions or j in conflict_positions:
            child[i], child[j] = swap_orientations(e, parent[j], parent[i])

            # 2-swap between 2 randomly selected locations in the corners
            valid_ids = [0, b - 1, e.n_piece - b, e.n_piece - 1]
            i, j = random.sample(valid_ids, 2)
            child[i], child[j] = swap_orientations(e, parent[j], parent[i])

    return child


def piece_type(piece):
    count_gray = piece.count(GRAY)
    return "corner" if count_gray == 2 else "edge" if count_gray == 1 else "inner"


def swap_orientations(e: EternityPuzzle, piece1, piece2):
    if piece_type(piece1) == "corner" and piece_type(piece2) == "corner":
        gray_positions_1 = [i for i in range(4) if piece1[i] == GRAY]
        gray_positions_2 = [i for i in range(4) if piece2[i] == GRAY]

        for rotated_piece1 in e.generate_rotation(piece1):
            if (
                rotated_piece1[gray_positions_2[0]] == GRAY
                and rotated_piece1[gray_positions_2[1]] == GRAY
            ):
                piece1 = rotated_piece1

        for rotated_piece2 in e.generate_rotation(piece2):
            if (
                rotated_piece2[gray_positions_1[0]] == GRAY
                and rotated_piece2[gray_positions_1[1]] == GRAY
            ):
                piece1 = rotated_piece2

    if piece_type(piece1) == "edge" and piece_type(piece2) == "edge":
        gray_position_1 = [i for i in range(4) if piece1[i] == GRAY][0]
        gray_position_2 = [i for i in range(4) if piece2[i] == GRAY][0]

        for rotated_piece1 in e.generate_rotation(piece1):
            if rotated_piece1[gray_position_2] == GRAY:
                piece1 = rotated_piece1

        for rotated_piece2 in e.generate_rotation(piece2):
            if rotated_piece2[gray_position_1] == GRAY:
                piece2 = rotated_piece2

    return piece1, piece2


def get_conflict_positions(e, solution):
    positions = []

    for j in range(e.board_size):
        for i in range(e.board_size):
            k = e.board_size * j + i
            k_east = e.board_size * j + (i - 1)
            k_south = e.board_size * (j - 1) + i

            if i > 0 and solution[k][WEST] != solution[k_east][EAST]:
                positions.append(k)
                positions.append(k_east)

            if i == 0 and solution[k][WEST] != GRAY:
                positions.append(k)

            if i == e.board_size - 1 and solution[k][EAST] != GRAY:
                positions.append(k)

            if j > 0 and solution[k][SOUTH] != solution[k_south][NORTH]:
                positions.append(k)
                positions.append(k_south)

            if j == 0 and solution[k][SOUTH] != GRAY:
                positions.append(k)

            if j == e.board_size - 1 and solution[k][NORTH] != GRAY:
                positions.append(k)

    positions = list(set(positions))
    return positions
