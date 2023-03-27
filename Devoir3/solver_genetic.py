from typing import List, Tuple
from rcpsp import RCPSP
from math import ceil, floor
import time
import random
import numpy as np
import networkx as nx


def solve(rcpsp: RCPSP) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        rcpsp (RCPSP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    print(rcpsp.graph.nodes)

    time_limit = 5 * 60  # 20 * 60

    mutation_rate = 0.1
    pop_size = 300
    tournament_size = ceil(pop_size / 20)
    tournament_accepted = ceil(tournament_size / 5)
    num_generations = 1000
    no_progress_generations = 50
    max_m_crossover = 10

    return genetic_algorithm(
        rcpsp,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        max_m_crossover=max_m_crossover,
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
    max_m_crossover,
    mutation_rate,
    tournament_size,
    tournament_accepted,
    pop_size,
    time_limit,
):

    start_time = time.time()
    best_valid_solution = None

    improvement_timer = 0
    while time.time() - start_time < time_limit:

        # Generate the initial population
        population = generate_population(r, pop_size)

        # Iterate over the generations
        for _ in range(num_generations):

            # The parents selected for the next generation are the whole population

            # Create the offspring for the next generation
            offspring = []
            for j in range(len(population) // 2):

                parent1 = population[j]
                parent2 = population[len(population) - j - 1]

                child1, child2 = pmx_crossover(parent1, parent2)

                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)
                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = selection(
                r,
                population + offspring,
                pop_size,
                tournament_size,
                tournament_accepted,
            )

            # Update the best solution found so far
            fitness_scores = [fitness(r, s) for s in population]
            id_fittest = np.argmax(fitness_scores)
            fittest_solution = population[id_fittest]
            fittest_score = fitness_scores[id_fittest]

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
    print(solution)
    print(r.graph.nodes)
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
        earliest_start_time = max(
            solution[p] + r.graph.nodes[job_id]["duration"] for p in predecessors
        )

        # Add some randomness to the start time
        random_offset = random.choice(available_start_times[job_id])
        start_time = earliest_start_time + random_offset

        # Add the job to the solution
        solution[job_id] = start_time

        # Update the list of available start times for each job
        for successor in r.graph.successors(job_id):
            available_start_times[successor].append(
                start_time + r.graph.edges[job_id, successor]["weight"]
            )

    return solution


def generate_population(r: RCPSP, size):
    population = []
    for _ in range(size):
        population.append(generate_chromosome(r))
    return population


def pmx_crossover(parent1, parent2):
    """Perform Partially Matched Crossover (PMX) between two parent solutions.

    Args:
        parent1 (dict): a solution mapping job IDs to their start times
        parent2 (dict): a solution mapping job IDs to their start times

    Returns:
        A new solution generated by the PMX crossover of the parents.
    """
    # Choose two random crossover points
    print(parent1)
    print(parent2)

    n = len(parent1)
    cxpoint1 = random.randint(1, n - 1)
    cxpoint2 = random.randint(1, n - 1)
    if cxpoint2 < cxpoint1:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Copy the section between the two crossover points from parent 1 to the offspring
    offspring = {}
    for job_id in parent1:
        if cxpoint1 <= job_id <= cxpoint2:
            offspring[job_id] = parent1[job_id]

    # Copy missing jobs from parent 2 and find the corresponding position in parent 1
    for job_id in parent2:
        if job_id not in offspring:
            pos = job_id
            while pos in parent2 and pos not in offspring:
                pos = parent2[pos]
            offspring[job_id] = parent2[pos]

    return offspring


def mutation(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        # Select two random points to swap
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome


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
