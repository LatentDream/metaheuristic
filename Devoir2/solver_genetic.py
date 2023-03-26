from typing import List
from tsptw import TSPTW
import time
import random
import numpy as np
from math import ceil, floor
import random
from utils.ant import Ant
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

    time_limit = 30 * 60

    mutation_rate = 0.1
    pop_size = 300
    tournament_size = ceil(pop_size / 20)
    tournament_accepted = ceil(tournament_size / 5)
    num_generations = 1000
    no_progress_generations = 50
    max_m_crossover = 10

    global ant
    ant = Ant(tsptw)

    global pbs
    pbs = ProbabilisticBeamSearch(tsptw, ant, mu=1)

    return genetic_algorithm(
        tsptw,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        max_m_crossover=max_m_crossover,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=time_limit,
    )


# Genetic algorithm
def genetic_algorithm(
    tsptw: TSPTW,
    num_generations,
    no_progress_generations,
    max_m_crossover,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
):

    start_time = time.time()
    best_valid_solution = None

    # Calculate a good initial best_solution
    greedy_solution = greedy_tsp(tsptw)
    if greedy_solution and tsptw.verify_solution(greedy_solution):
        best_valid_solution = greedy_solution
        best_solution = best_valid_solution
        best_fitness = fitness(tsptw, best_solution)
        print(best_fitness)
        best_cost = tsptw.get_solution_cost(best_solution)
        print("Greedy solution cost :", best_cost)
        print("Greedy solution path :", best_solution)
    else:
        best_solution = pbs.beam_construct()
        best_fitness = fitness(tsptw, best_solution)
        best_cost = tsptw.get_solution_cost(best_solution)

    improvement_timer = 0
    while time.time() - start_time < time_limit:

        # Generate the initial population
        population = generate_population(tsptw, pop_size)

        # Iterate over the generations
        for _ in range(num_generations):

            # The parents selected for the next generation are the whole population

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(population) // 2):

                parent1 = population[j]
                parent2 = population[len(population) - j - 1]

                # Select the m value for m-point-crossover
                if tsptw.num_nodes <= 10:
                    m = 2
                else:
                    m = random.choice(range(2, max_m_crossover + 1))
                child1, child2 = m_point_crossover(
                    parent1,
                    parent2,
                    m=m,
                )

                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
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
            fitness_scores = [fitness(tsptw, s) for s in population]
            id_fittest = np.argmax(fitness_scores)
            fittest_solution = population[id_fittest]
            fittest_score = fitness_scores[id_fittest]

            if fittest_score > best_fitness:
                best_solution = fittest_solution
                best_fitness = fittest_score
                improvement_timer = 0
                if best_fitness > -10e9:
                    print(
                        "Best valid solution found : Cost = {}".format(
                            tsptw.get_solution_cost(best_solution)
                        )
                    )
                    best_valid_solution = best_solution
                else:
                    num_conflicts = floor(-best_fitness / 10e10)
                    if num_conflicts > 0:
                        print("Number of conflicts : ", floor(-best_fitness / 10e10))
            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer % no_progress_generations == 0:
                    break

    # If a valid solution has been found
    if best_valid_solution:
        print("Cost", tsptw.get_solution_cost(best_valid_solution))
        return best_valid_solution
    # If no valid solution has been found
    else:
        print("Number of violations", get_number_of_violations(best_solution, tsptw))
        print("Cost", tsptw.get_solution_cost(best_solution))
        return best_solution


###################################### Evaluation Functions ####################################


def fitness(tsptw: TSPTW, solution):
    return -(
        10e10 * get_number_of_violations(solution, tsptw)
        + tsptw.get_solution_cost(solution)
    )


def get_number_of_violations(solution: List[int], tsptw: TSPTW) -> int:
    nb_of_violation = 0
    time_step = 0
    last_stop = 0
    for next_stop in solution[1:]:
        edge = (last_stop, next_stop)
        time_step += tsptw.graph.edges[edge]["weight"]
        time_windows_begin, time_windows_end = tsptw.time_windows[next_stop]
        if time_step < time_windows_begin:
            waiting_time = time_windows_begin - time_step
            time_step += waiting_time
        if time_step > time_windows_end:
            nb_of_violation += 1

    return nb_of_violation


############################################# Genetic Operations ###############################

# Generation of the population
def generate_population(tsptw, pop_size):
    population = []
    for _ in range(pop_size):
        population.append(generate_chromosome(tsptw))
    return population


# Generate a single random solution
def generate_chromosome(tsptw: TSPTW):
    chromosome = list(range(1, tsptw.num_nodes))
    random.shuffle(chromosome)
    return [0] + chromosome + [0]


# Selection through a tournament
def selection(tsptw: TSPTW, population, pop_size, tournament_size, tournament_accepted):
    selected = []
    while len(selected) < pop_size:
        subset = random.sample(population, tournament_size)
        selected.extend(
            sorted(subset, key=lambda s: fitness(tsptw, s), reverse=True)[
                :tournament_accepted
            ]
        )
    return selected


# Crossover function
def m_point_crossover(parent1, parent2, m):

    parent1 = parent1[1:-1]
    parent2 = parent2[1:-1]

    # Ensure m is between 0 and len(parent1) - 1
    m = max(min(m, len(parent1) - 1), 0)

    # Choose m random crossover points
    crossover_points = random.sample(range(1, len(parent1)), m)

    # Sort the crossover points
    crossover_points.sort()

    # Initialize offspring chromosomes
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Perform crossover between the parents at the crossover points
    for i in range(0, m, 2):
        start_point = crossover_points[i]
        if i + 1 < m:
            end_point = crossover_points[i + 1]
        else:
            end_point = len(parent1)

        # Swap the corresponding segments of the parents
        offspring1[start_point:end_point], offspring2[start_point:end_point] = (
            offspring2[start_point:end_point],
            offspring1[start_point:end_point],
        )

    # Repair offspring chromosomes by replacing repeated elements with missing elements
    offspring1 = repair_chromosome(offspring1, parent1)
    offspring2 = repair_chromosome(offspring2, parent2)

    offspring1 = [0] + offspring1 + [0]
    offspring2 = [0] + offspring2 + [0]

    return offspring1, offspring2


def repair_chromosome(chromosome, parent1):
    # Get the set of missing elements in the offspring chromosome
    missing_elements = list(set(parent1) - set(chromosome))

    while len(missing_elements) > 0:
        # Get the set of repeated elements in the offspring chromosome
        repeated_elements = set([x for x in chromosome if chromosome.count(x) > 1])
        # Replace repeated elements with missing elements
        for i in range(len(chromosome)):
            if chromosome[i] in repeated_elements:
                chromosome[i] = missing_elements[0]
                missing_elements = missing_elements[1:]
                repeated_elements = set(
                    [x for x in chromosome if chromosome.count(x) > 1]
                )
                missing_elements = list(set(parent1) - set(chromosome))
                if len(missing_elements) == 0:
                    break
    return chromosome


# Mutation function
def mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        # Select two random points to swap
        idx1, idx2 = random.sample(range(1, len(chromosome) - 1), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


############################# Greedy Solver ###################################################


def greedy_tsp(tsptw: TSPTW):
    time_constraints = tsptw.time_windows
    nodes = list(range(tsptw.num_nodes))
    nodes = sorted(nodes, key=lambda x: time_constraints[x][0])

    current_time = 0
    solution = []
    remaining_nodes = set(nodes)

    # Find the node with the earliest opening time
    first_node = nodes[0]
    solution.append(first_node)

    remaining_nodes.remove(first_node)

    # Explore the remaining nodes in the order of their opening windows
    while remaining_nodes:

        next_node = None
        for node in sorted(remaining_nodes, key=lambda x: time_constraints[x][0]):
            if check_time_constraint(
                tsptw, solution[-1], node, current_time, time_constraints
            ):
                next_node = node
                break

        if next_node is None:
            # If no remaining node satisfies the time constraint, return None
            return None
        else:
            solution.append(next_node)
            remaining_nodes.remove(next_node)
            current_time = max(
                current_time + calculate_distance(tsptw, solution[-2], next_node),
                time_constraints[next_node][0],
            )

    solution.append(0)
    return solution


def check_time_constraint(tsptw: TSPTW, node1, node2, timer, time_constraints):
    # Check if the time window of node2 is valid given that node1 was visited at the start of its time window
    travel_time = tsptw.graph[node1][node2]["weight"]
    arrival_time = max(timer + travel_time, time_constraints[node2][0])
    return arrival_time <= time_constraints[node2][1]


def calculate_distance(tsptw, node1, node2):
    return tsptw.graph[node1][node2]["weight"]
