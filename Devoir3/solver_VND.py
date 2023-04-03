from copy import copy
from typing import Dict, List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np
from utils.utils import generate_random_valid_solution


def VND(r: RCPSP, time_limit=10*60):

    start_time = time()
    tic = start_time

    with tqdm(total=time_limit) as progress_bar:

        # Initial solution
        solution = generate_random_valid_solution(r)
        print(f"Solution initial: {solution}")
        # Build the neighborhood priority list (priority to node that have the most sucessors)
        nodes = sorted(list(r.graph.nodes))
        del nodes[0]
        node_neighbor = sorted([(node_id, len(list(r.graph.successors(node_id)))) for node_id in nodes], key=lambda x: -x[1])
        node_neighbor = [element[0] for element in node_neighbor]

        # Parameter
        k = 4
        k_max = len(node_neighbor) - 1

        while k != k_max:
        
            new_solution = local_search(r, solution, k, node_neighbor)
            print(f"New Solution initial: {solution}")

            solution, k = neighborhood_change(r, solution, new_solution, k)
            
            if (tac:=time()) - start_time < time_limit:
                progress_bar.update(tac - tic)
                tic = tac
            else:
                print("\nTime out - Returning current best\n")
                return solution

    return solution


def neighborhood_change(r: RCPSP, solution: Dict[int, int], new_solution: Dict[int, int], k: int):
    if r.get_solution_cost(new_solution) < r.get_solution_cost(solution):
       return new_solution, k
    else:
        return solution, k+1


def local_search(r: RCPSP, solution: Dict[int, int], k: int, job_neighbor: list):
    neighbor_job_id = job_neighbor[k]
    original_neighbor_starting_time = solution[neighbor_job_id]
    neighbor_ressource_needed = np.array(r.graph.nodes[neighbor_job_id]["resources"])
    neighbor_job_duration = r.graph.nodes[neighbor_job_id]["duration"]

    # Build solution up to job k
    new_solution = {}
    time_step_to_job_id = {}
    for job_id in solution.keys():
        if (job_starting_time:=solution[job_id]) < original_neighbor_starting_time:
            new_solution[job_id] = job_starting_time
            if job_starting_time in time_step_to_job_id.keys():
                time_step_to_job_id[job_starting_time].append(job_id)
            else:
                time_step_to_job_id[job_starting_time] = [job_id]

    # Init search params
    current_time = 0
    available_ressources = np.array(r.resource_availabilities)
    available_ressources_through_time = [copy(available_ressources) for _ in range(original_neighbor_starting_time)]
    jobs_done_through_time = [set() for _ in range(original_neighbor_starting_time)]


    # Step #1: Find which ressources are available and which jobs is done at each timestep until neighbor_starting_time
    while current_time < original_neighbor_starting_time:
        if current_time in time_step_to_job_id.keys():
            for curr_job_id in time_step_to_job_id[current_time]:
                curr_job_ressource = np.array(r.graph.nodes[curr_job_id]["resources"])
                duration = r.graph.nodes[curr_job_id]["duration"]
                locked_until = current_time + duration
                # remove the ressources for the curr_job
                for i in range(current_time, min(original_neighbor_starting_time, locked_until)):
                    available_ressources_through_time[i] -= curr_job_ressource
                # Save at which time step the job is completed
                if locked_until < original_neighbor_starting_time:
                    jobs_done_through_time[locked_until].add(curr_job_id)
        current_time += 1
    job_done = set()
    for i in range(len(jobs_done_through_time)):
        jobs_done_through_time[i].update(job_done)
        job_done.update(jobs_done_through_time[i])


    # Step #2: Try to insert the neighbor_job_id before it's inital starting time without moving the other node
    while current_time < original_neighbor_starting_time:
        # Check the precedence constraints
        if set(r.graph.predecessors(neighbor_job_id)).issubset(jobs_done_through_time[current_time]):
            # Check if there is enough ressource available
            ressource_needed_util = current_time + neighbor_job_duration
            can_be_inserted_here = True
            for i in range(current_time, min(original_neighbor_starting_time, ressource_needed_util)):
                can_be_inserted_here &= min(available_ressources_through_time[i] - neighbor_ressource_needed) > 0
            if can_be_inserted_here:
                new_solution[neighbor_job_id] = current_time
                if current_time in time_step_to_job_id.keys():
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
    current_running_job_blocks = set()
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
            current_running_job_blocks.difference_update(set(new_available_job))

        if current_time in time_step_to_job_id.keys():
            # Launch the jobs
            for job_id in time_step_to_job_id[current_time]:
                ressources_needed_to_start_job = np.array(r.graph.nodes[job_id]["resources"])
                duration = r.graph.nodes[job_id]["duration"]
                next_available_job = list(r.graph.successors(job_id))
                # Locked ressource
                if duration > 0:
                    locked_until = current_time + duration  
                    available_ressources -= ressources_needed_to_start_job
                    if locked_until in locked_ressources.keys():
                        locked_ressources[locked_until] += ressources_needed_to_start_job
                    else:
                        locked_ressources[locked_until] = ressources_needed_to_start_job
                    if locked_until in next_job.keys():
                        next_job[locked_until] += next_available_job
                        current_running_job_blocks.update(next_available_job)
                    else:
                        next_job[locked_until] = next_available_job
                        current_running_job_blocks.update(next_available_job)
                else:
                    available_job += next_available_job
                del available_job[available_job.index(job_id)]
        current_time += 1

    print(new_solution)

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
            current_running_job_blocks.difference_update(set(new_available_job))

        for _ in range(max(int(len(available_job) / 2), 1)):
            job_idx = random.randrange(0, len(available_job))
            job_id = available_job[job_idx]
            if job_id in new_solution.keys():
                del available_job[job_idx]
                continue
            ressources_needed_to_start_job = np.array(r.graph.nodes[job_id]["resources"])

            # Check the precedence constraints
            precedent_jobs = set(r.graph.predecessors(job_id))
            constraints_violated = not precedent_jobs.issubset(new_solution.keys())
            constraints_violated |= bool(precedent_jobs.intersection(current_running_job_blocks))
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
                        locked_ressources[locked_until] += ressources_needed_to_start_job
                    else:
                        locked_ressources[locked_until] = ressources_needed_to_start_job
                    if locked_until in next_job.keys():
                        next_job[locked_until] += next_available_job
                    else:
                        next_job[locked_until] = next_available_job
                else:
                    available_job += next_available_job
        
        current_time += 1



    return new_solution


