from copy import copy
from typing import Dict, List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np
from utils.tree_node import TreeNode
from utils.utils import (
    generate_random_valid_solution,
    find_ressources_used_at_each_timestep,
)


def solve(rcpsp: RCPSP):

    time_limit = 10  # 20 * 60

    # return VND(rcpsp, time_limit, build_neighbor_priority_with_stochasticity)
    return VND(rcpsp, time_limit, build_neighbor_priority)


def VND(r: RCPSP, time_limit, build_neighbor):

    start_time = time()
    tic = start_time

    with tqdm(total=time_limit) as progress_bar:

        best_solution = generate_random_valid_solution(r)

        while True:

            # Initial solution
            solution = generate_random_valid_solution(r)
            node_neighbor = build_neighbor(r)

            # Parameter : Neighborhood structure
            k = 0
            k_max = len(node_neighbor) - 1

            while k != k_max:
                new_solution = local_search(r, solution, k, node_neighbor)
                optimised_solution = optimise(r, new_solution)
                solution, k = neighborhood_change(r, solution, optimised_solution, k)

            if r.verify_solution(solution):
                if r.get_solution_cost(solution) < r.get_solution_cost(best_solution):
                    best_solution = copy(solution)
            else:
                print("/!\ Warning: invalid solution detected /!\ ")

            if (tac := time()) - start_time < time_limit:
                progress_bar.update(tac - tic)
                tic = tac
            # else:
            #     break

    return best_solution


def build_random_neighbor_priority(r: RCPSP):
    nodes = [i + 1 for i in range(1, r.graph.number_of_nodes())]
    priority_neighbord = []
    for _ in range(len(nodes)):
        nodes_idx = random.randrange(0, len(nodes))
        priority_neighbord.append(nodes[nodes_idx])
        del nodes[nodes_idx]
    return priority_neighbord


def build_neighbor_priority_with_stochasticity(r: RCPSP):
    # Build the neighborhood priority list (priority to node that have the most sucessors)
    nodes = [i + 1 for i in range(1, r.graph.number_of_nodes())]
    node_neighbor = sorted(
        [
            (
                node_id,
                len(list(r.graph.successors(node_id))) + np.random.uniform(0.0, 0.5),
            )
            for node_id in nodes
        ],
        key=lambda x: -x[1],
    )
    node_neighbor = [element[0] for element in node_neighbor]
    return node_neighbor


def build_neighbor_priority(r: RCPSP):
    # Build a acyclic graph representaiton of the job dependencies
    nodes = {i: TreeNode(i) for i in range(1, r.graph.number_of_nodes() + 1)}
    for i in range(1, r.graph.number_of_nodes() + 1):
        children_id = r.graph.successors(i)
        nodes[i].add_children(*[nodes[child_id] for child_id in children_id])
    # fin number of job waiting for the job_i to be done
    node_priority = [
        (i, nodes[i].depth()) for i in range(1, r.graph.number_of_nodes() + 1)
    ]
    node_neighbor = sorted(node_priority, key=lambda x: x[1])
    return [element[0] for element in node_neighbor]


def neighborhood_change(
    r: RCPSP, solution: Dict[int, int], new_solution: Dict[int, int], k: int
):
    if r.get_solution_cost(new_solution) < r.get_solution_cost(solution):
        return new_solution, k
    else:
        return solution, k + 1


def local_search(r: RCPSP, solution: Dict[int, int], k: int, job_neighbor: list):
    neighbor_job_id = job_neighbor[k]
    original_neighbor_starting_time = solution[neighbor_job_id]
    neighbor_ressource_needed = np.array(r.graph.nodes[neighbor_job_id]["resources"])
    neighbor_job_duration = r.graph.nodes[neighbor_job_id]["duration"]

    # Build solution up to job k
    new_solution = {}
    time_step_to_job_id = {}
    for job_id in range(1, len(solution.keys()) + 1):
        if (
            job_starting_time := solution[job_id]
        ) <= original_neighbor_starting_time and job_id != neighbor_job_id:
            new_solution[job_id] = job_starting_time
            if job_starting_time in time_step_to_job_id.keys():
                time_step_to_job_id[job_starting_time].append(job_id)
            else:
                time_step_to_job_id[job_starting_time] = [job_id]

    # Init search params
    current_time = 0
    available_ressources = np.array(r.resource_availabilities)
    available_ressources_through_time = [
        copy(available_ressources) for _ in range(original_neighbor_starting_time)
    ]
    jobs_done_through_time = [set() for _ in range(original_neighbor_starting_time + 1)]

    # Step #1: Find which ressources are available and which jobs is done at each timestep until neighbor_starting_time
    while current_time < original_neighbor_starting_time:
        if current_time in time_step_to_job_id.keys():
            for curr_job_id in time_step_to_job_id[current_time]:
                curr_job_ressource = np.array(r.graph.nodes[curr_job_id]["resources"])
                duration = r.graph.nodes[curr_job_id]["duration"]
                locked_until = current_time + duration
                # remove the ressources for the curr_job
                for i in range(
                    current_time, min(original_neighbor_starting_time, locked_until)
                ):
                    available_ressources_through_time[i] -= curr_job_ressource
                # Save at which time step the job is completed
                if locked_until < original_neighbor_starting_time:
                    jobs_done_through_time[locked_until].add(curr_job_id)
        current_time += 1
    job_done = set()
    for i in range(len(jobs_done_through_time)):
        jobs_done_through_time[i].update(job_done)
        job_done.update(jobs_done_through_time[i])

    current_time = 0
    # Step #2: Try to insert the neighbor_job_id before it's inital starting time without moving the other node
    while current_time <= original_neighbor_starting_time:
        # Check the precedence constraints
        if set(r.graph.predecessors(neighbor_job_id)).issubset(
            jobs_done_through_time[current_time]
        ):
            # Check if there is enough ressource available
            ressource_needed_util = current_time + neighbor_job_duration
            can_be_inserted_here = True
            for i in range(
                current_time,
                min(original_neighbor_starting_time, ressource_needed_util),
            ):
                can_be_inserted_here &= (
                    min(
                        available_ressources_through_time[i] - neighbor_ressource_needed
                    )
                    > 0
                )
            if can_be_inserted_here:
                new_solution[neighbor_job_id] = current_time
                if current_time in time_step_to_job_id.keys():
                    if neighbor_job_duration == 0:
                        time_step_to_job_id[current_time].insert(0, neighbor_job_id)
                    else:
                        time_step_to_job_id[current_time].append(neighbor_job_id)
                else:
                    time_step_to_job_id[current_time] = [neighbor_job_id]
                break
        current_time += 1
    # use the opriginal time if it can't be inserted hearlier
    if not neighbor_job_id in new_solution.keys():
        new_solution[neighbor_job_id] = original_neighbor_starting_time
        if original_neighbor_starting_time in time_step_to_job_id.keys():
            time_step_to_job_id[original_neighbor_starting_time].append(neighbor_job_id)
        else:
            time_step_to_job_id[original_neighbor_starting_time] = [neighbor_job_id]

    # Step #3: Add the rest of the node randomly
    ## Build the new_solution
    current_time = 0
    available_job = [1]
    available_ressources = np.array(r.resource_availabilities)
    locked_ressources = dict()
    next_job = dict()
    # Do all the jobs that are already in new_solution
    while current_time <= original_neighbor_starting_time:

        # Free newly available ressource
        if current_time in locked_ressources.keys():
            available_ressources += locked_ressources[current_time]
            del locked_ressources[current_time]
        # Add new job that can be done
        if current_time in next_job.keys():
            new_available_job = next_job[current_time]
            available_job += new_available_job
            del next_job[current_time]

        if current_time in time_step_to_job_id.keys():
            # Launch the jobs
            for job_id in time_step_to_job_id[current_time]:
                ressources_needed_to_start_job = np.array(
                    r.graph.nodes[job_id]["resources"]
                )
                duration = r.graph.nodes[job_id]["duration"]
                next_available_job = list(r.graph.successors(job_id))
                # Locked ressource
                if duration > 0:
                    locked_until = current_time + duration
                    available_ressources -= ressources_needed_to_start_job
                    if locked_until in locked_ressources.keys():
                        locked_ressources[
                            locked_until
                        ] += ressources_needed_to_start_job
                    else:
                        locked_ressources[locked_until] = ressources_needed_to_start_job
                    if locked_until in next_job.keys():
                        next_job[locked_until] += next_available_job
                    else:
                        next_job[locked_until] = next_available_job
                else:
                    available_job += next_available_job
                del available_job[available_job.index(job_id)]
        current_time += 1

    # Do the other job
    while len(new_solution.keys()) != len(solution.keys()):

        # Free newly available ressource
        if current_time in locked_ressources.keys():
            available_ressources += locked_ressources[current_time]
            del locked_ressources[current_time]
        # Add new job that can be done
        if current_time in next_job.keys():
            new_available_job = next_job[current_time]
            available_job += new_available_job
            del next_job[current_time]

        if len(available_job) > 0:
            for _ in range(max(int(len(available_job) / 2), 1)):
                job_idx = random.randrange(0, len(available_job))
                job_id = available_job[job_idx]
                if job_id in new_solution.keys():
                    del available_job[job_idx]
                    continue
                ressources_needed_to_start_job = np.array(
                    r.graph.nodes[job_id]["resources"]
                )

                # Check the precedence constraints
                precedent_jobs = set(r.graph.predecessors(job_id))
                constraints_violated = not all(
                    [
                        job in new_solution.keys()
                        and new_solution[job] + r.graph.nodes[job]["duration"]
                        < current_time
                        for job in precedent_jobs
                    ]
                )
                if constraints_violated:
                    continue

                # Launch the job
                if min(available_ressources - ressources_needed_to_start_job) >= 0:
                    new_solution[job_id] = current_time
                    duration = r.graph.nodes[job_id]["duration"]
                    next_available_job = list(r.graph.successors(job_id))
                    del available_job[job_idx]
                    # Lock the ressource
                    if duration > 0:
                        locked_until = current_time + duration
                        available_ressources -= ressources_needed_to_start_job
                        if locked_until in locked_ressources.keys():
                            locked_ressources[
                                locked_until
                            ] += ressources_needed_to_start_job
                        else:
                            locked_ressources[
                                locked_until
                            ] = ressources_needed_to_start_job
                        if locked_until in next_job.keys():
                            next_job[locked_until] += next_available_job
                        else:
                            next_job[locked_until] = next_available_job

                    else:
                        available_job += next_available_job

        current_time += 1

    return new_solution


def optimise(r: RCPSP, solution: Dict[int, int]):

    # Find which task is starting at each timestep (reverse the dictionnary)
    time_to_job = {}
    for job in solution.keys():
        if solution[job] in time_to_job:
            if r.graph.nodes[job]["duration"] == 0:
                time_to_job[solution[job]].insert(0, job)
            else:
                time_to_job[solution[job]].append(job)
        else:
            time_to_job[solution[job]] = [job]

    # Squeeze the most at the of the
    current_time = 0
    optimised_solution = {}

    while len(optimised_solution.keys()) != len(solution.keys()):

        if current_time in time_to_job:
            for job in time_to_job[current_time]:
                # Try to start the job earlier
                insert_time = current_time - 1
                earliest_insert_time_found = current_time
                available_ressources_through_time = (
                    find_ressources_used_at_each_timestep(
                        r, optimised_solution, current_time
                    )
                )
                job_ressources = r.graph.nodes[job]["resources"]
                while insert_time >= 0:

                    # Check the precedence constraint
                    constraint_violated = False
                    precendent_jobs = r.graph.predecessors(job)
                    for precendent_job in precendent_jobs:
                        duration = r.graph.nodes[precendent_job]["duration"]
                        constraint_violated |= (
                            optimised_solution[precendent_job] + duration > insert_time
                        )  # Todo: Verif if's right
                    if constraint_violated:
                        break

                    # Check ressources availability
                    if (
                        min(
                            available_ressources_through_time[insert_time]
                            - job_ressources
                        )
                        >= 0
                    ):
                        earliest_insert_time_found = insert_time
                    else:
                        break

                    insert_time -= 1

                optimised_solution[job] = earliest_insert_time_found

        current_time += 1

    return optimised_solution
