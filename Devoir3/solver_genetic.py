from typing import List
from rcpsp import RCPSP
from math import ceil, floor
import time
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from math import inf


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

    mutation_rate = 0.005
    pop_size = 30
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 20
    elite_size = 2

    return genetic_algorithm(
        rcpsp,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        tournament_accepted=tournament_accepted,
        pop_size=pop_size,
        time_limit=time_limit,
    )


def genetic_algorithm(
    r: RCPSP,
    num_generations,
    no_progress_generations,
    elite_size,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
):

    start_time = time.time()
    best_valid_solution = None
    best_fitness = -inf
    improvement_timer = 0
    time_over = False
    while not time_over:

        # Generate the initial population
        population = generate_population(r, pop_size, elite_size=elite_size)
        # Iterate over the generations
        for _ in range(num_generations):

            elite = sorted(population, key=lambda s: fitness(r, s), reverse=True)[
                :elite_size
            ]

            # The parents selected for the next generation are the whole population
            population = sorted(population, key=lambda s: fitness(r, s), reverse=True)[
                : pop_size // 2
            ]

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(population) // 2):

                parent1 = population[j]
                parent2 = population[len(population) - j - 1]

                child1, child2 = crossoverPMX(parent1, parent2)

                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)

                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = selection(
                r,
                population + offspring + elite,
                pop_size,
                tournament_size,
                tournament_accepted,
            )

            # Update the best solution found so far
            fitness_scores = [fitness(r, s) for s in population]
            id_fittest = np.argmax(fitness_scores)
            fittest_solution = population[id_fittest]
            fittest_score = fitness_scores[id_fittest]
            # print(fittest_solution)
            print("Conflicts : ", max(0, -floor(fittest_score)))
            if fittest_score > best_fitness:
                best_solution = fittest_solution
                best_fitness = fittest_score
                improvement_timer = 0
                if best_fitness > 0:
                    print(
                        "Best valid solution found : Cost = {}".format(
                            r.get_solution_cost(best_solution)
                        )
                    )
                    best_valid_solution = best_solution
            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if improvement_timer % no_progress_generations == 0:
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    # If a valid solution has been found
    if best_valid_solution:
        print("Cost", r.get_solution_cost(best_valid_solution))
        print("Solution valid ", r.verify_solution(best_fitness))
        return best_valid_solution
    else:
        return best_solution


###################################### Evaluation Functions ####################################


def fitness(r: RCPSP, solution):

    st_conflict = start_conflict(r, solution)
    prec_conflicts = precedence_conflicts(r, solution)
    res_conflicts = resource_conflicts(r, solution)

    fitness = 1 / r.get_solution_cost(solution) - (
        st_conflict + prec_conflicts + res_conflicts
    )

    return fitness


def start_conflict(r: RCPSP, solution):
    min_start_time = min([solution[job] for job in r.graph.nodes])
    if min_start_time != 0:
        return 1
    return 0


def precedence_conflicts(r: RCPSP, solution):
    n_conflicts = 0
    # Check precedence constraints
    for job in r.graph.nodes:
        duration = r.graph.nodes[job]["duration"]
        job_start_time = solution[job]
        job_finish_time = job_start_time + duration
        for successor in r.graph.successors(job):
            if solution[successor] < job_finish_time:
                n_conflicts += 1
    return n_conflicts


def resource_conflicts(r: RCPSP, solution):
    n_conflicts = 0
    # Check resource constraints
    num_resources = len(r.resource_availabilities)

    # Find the maximum finish time to set the range for resource usage check
    max_finish_time = max(
        [solution[job] + r.graph.nodes[job]["duration"] for job in r.graph.nodes]
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


###################################### Genetic Operaters ###############################


def generate_chromosome(r: RCPSP):
    horizon = 181
    solution = {}

    # Sort tasks by their order in the graph, assuming the tasks are numbered sequentially
    tasks = list(r.graph.nodes)

    for task in tasks:
        solution[task] = random.choice([i for i in range(0, horizon + 1)])

    set_start_time_zero(r, solution)

    return solution


def generate_fit_chromosome(r):
    solution = {}

    # Set the start time of the first job to 0
    first_job = 1
    solution[first_job] = 0

    # Initialize the list of available start times for each job
    available_start_times = {job_id: [0] for job_id in r.graph.nodes}

    # Iterate over the jobs in topological order
    for job_id in nx.topological_sort(r.graph):
        if job_id == first_job:
            continue

        # Get the list of predecessors of the current job
        predecessors = list(r.graph.predecessors(job_id))

        # Compute the earliest start time for the current job
        earliest_start_time = r.graph.nodes[job_id]["duration"] + max(
            solution[p] for p in predecessors
        )

        # Add some randomness to the start time
        random_offset = random.choice(available_start_times[job_id])
        start_time = earliest_start_time + random_offset

        # Add the job to the solution
        solution[job_id] = start_time

        # Update the list of available start times for each job
        for successor in r.graph.successors(job_id):
            weight = r.graph.edges[job_id, successor].get("weight", 0)
            available_start_times[successor].append(start_time + weight)

    set_start_time_zero(r, solution)

    return solution


def generate_population(r: RCPSP, pop_size, elite_size):
    population = []

    for _ in range(elite_size):
        population.append(generate_fit_chromosome(r))
    for _ in range(pop_size):
        population.append(generate_chromosome(r))
    return population


def crossoverPMX(parent1, parent2):
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    keys = list(parent1.keys())
    # random.shuffle(keys)
    # keys.sort()
    xpoint11, xpoint21 = getXPoints(len(keys))
    xpoint12, xpoint22 = getXPoints(len(keys))

    for i in range(xpoint11 - 1, xpoint21 - 1):
        child1[keys[i]] = parent2[keys[i]]

    for i in range(xpoint12 - 1, xpoint22 - 1):
        child2[keys[i]] = parent1[keys[i]]

    return child1, child2


def getXPoints(length):
    gene1 = random.randint(0, length)
    gene2 = random.randint(0, length)
    xpoint1 = min(gene1, gene2)
    xpoint2 = max(gene1, gene2)
    if xpoint2 == xpoint1:
        if xpoint2 < length:
            xpoint2 += 1
        else:
            xpoint1 -= 1
    return xpoint1, xpoint2


def mutation(solution, mutation_rate):
    mutated_solution = deepcopy(solution)

    for job_id in solution.keys():
        if random.random() < mutation_rate:
            mutated_solution[job_id] = random.randint(
                0, max(mutated_solution.values()) + 1
            )

    return mutated_solution


# Selection through a tournament
def selection(r: RCPSP, population, pop_size, tournament_size, tournament_accepted):

    selected = []
    while len(selected) < pop_size:
        subset = random.sample(population, tournament_size)
        selected.extend(
            sorted(subset, key=lambda s: fitness(r, s), reverse=True)[
                :tournament_accepted
            ]
        )
    return selected


def set_start_time_zero(r: RCPSP, solution):
    min_start_time = min([solution[job] for job in r.graph.nodes])
    if min_start_time > 0:
        key = [k for k, v in solution.items() if v == min_start_time][0]
        solution[key] = 0
        return solution
    else:
        return solution
