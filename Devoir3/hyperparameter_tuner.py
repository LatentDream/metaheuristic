from typing import List, Dict
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

    
    
    mutations_rates = [0.001 * (12 * i) for i in range(1, 6)]
    pop_sizes = [20 * i for i in range(1, 6)]
    tournament_sizes = [10 * i for i in range(1, 6)]
    tournament_accepted_sizes = [5 * i for i in range(1, 6)]
    
    time_limit = 5 * 60  # 20 * 60

    pop_size = 20
    mutation_rate = 1 / pop_size
    max_iter_local_search = 200
    tournament_size = 10
    tournament_accepted = 5
    num_generations = 100
    no_progress_generations = 20
    elite_size = 1

    return genetic_algorithm(
        rcpsp,
        num_generations=num_generations,
        no_progress_generations=no_progress_generations,
        max_iter_local_search=max_iter_local_search,
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
    max_iter_local_search,
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
        population = sorted(population, key=lambda s: fitness(r, s), reverse=True)

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

                child1, child2 = crossoverPMX(parent1, parent2)

                child1 = mutation(child1, mutation_rate)
                child2 = mutation(child2, mutation_rate)

                offspring.append(child1)
                offspring.append(child2)

            # Select the survivors for the next generation : we keep the same population size
            population = (
                selection(
                    r,
                    population + offspring,
                    pop_size,
                    tournament_size,
                    tournament_accepted,
                )
                + elite
            )

            population = sorted(population, key=lambda s: fitness(r, s), reverse=True)

            # Update the best solution found so far
            fittest_solution = population[0]
            fittest_score = fitness(r, fittest_solution)
            # print(fittest_solution)
            nb_conflicts = max(0, -floor(fittest_score))
            print("Conflicts : ", nb_conflicts)

            if fittest_score > best_fitness:
                fittest_solution = local_search(
                    r, fittest_solution, max_iterations=max_iter_local_search
                )

                best_solution = fittest_solution.copy()
                best_fitness = fitness(r, best_solution)
                improvement_timer = 0
                if best_fitness > 0:

                    print(
                        "Best valid solution found : Cost = {}".format(
                            r.get_solution_cost(best_solution)
                        )
                    )
                    best_valid_solution = best_solution.copy()
            else:
                improvement_timer += 1
                # If no improvement is made during too many generations, restart on a new population
                if (
                    improvement_timer > no_progress_generations == 0
                    and nb_conflicts > 2
                ):
                    improvement_timer = 0
                    break

            if time.time() - start_time > time_limit:
                time_over = True
                break

    # If a valid solution has been found
    if best_valid_solution:
        print("Cost", r.get_solution_cost(best_valid_solution))
        print("Solution valid ", r.verify_solution(best_valid_solution))
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


def set_start_time_zero(r: RCPSP, solution):
    min_start_time = min([solution[job] for job in r.graph.nodes])
    if min_start_time > 0:
        key = [k for k, v in solution.items() if v == min_start_time][0]
        solution[key] = 0
        return solution
    else:
        return solution


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


################################# Local Search to improve valid solutions ##############################################


def local_search(
    r: RCPSP, solution: Dict[int, int], max_iterations: int
) -> Dict[int, int]:

    horizon = 181

    current_solution = solution.copy()
    best_solution = solution.copy()
    best_fitness = fitness(r, best_solution)

    for _ in range(max_iterations):

        # Choose a random job and a new start time for it
        job_id = random.choice(list(r.graph.nodes))

        if len(list(r.graph.predecessors(job_id))) > 0:
            earliest_start_time = max(
                solution[p] + r.graph.nodes[job_id]["duration"]
                for p in r.graph.predecessors(job_id)
            )
        else:
            earliest_start_time = solution[job_id] - r.graph.nodes[job_id]["duration"]

        if len(list(r.graph.successors(job_id))) > 0:
            latest_start_time = min(
                solution[s] - r.graph.nodes[job_id]["duration"]
                for s in r.graph.successors(job_id)
            )
        else:
            latest_start_time = solution[job_id] + r.graph.nodes[job_id]["duration"]

        if latest_start_time < earliest_start_time:
            continue

        new_start_time = random.randint(earliest_start_time, latest_start_time)

        # Update the solution with the new start time
        current_solution[job_id] = new_start_time

        # Compute the fitness of the new solution and decide whether to accept it
        current_fitness = fitness(r, current_solution)
        if current_fitness < best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness

        # If the new solution is not accepted, revert the change
        else:
            current_solution = best_solution.copy()

    return best_solution
