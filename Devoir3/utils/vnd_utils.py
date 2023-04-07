
from copy import copy
from typing import Dict
from rcpsp import RCPSP
import random
import numpy as np
from utils.utils import find_ressources_used_at_each_timestep, randomdly_insert_missing_job
from math import inf


def is_precedence_constraint_violated(r: RCPSP, solution, job_id, current_time):
    # Check the precedence constraints
    precedent_jobs = set(r.graph.predecessors(job_id))      
    constraints_violated = not all([job in solution.keys() and solution[job] + r.graph.nodes[job]["duration"] < current_time for job in precedent_jobs])
    return constraints_violated


def local_search(r: RCPSP, solution: Dict[int, int], k: int, job_neighbor: list):
    neighbor_job_id = job_neighbor[k]
    original_neighbor_starting_time = solution[neighbor_job_id]
    neighbor_ressource_needed = np.array(r.graph.nodes[neighbor_job_id]["resources"])
    neighbor_job_duration = r.graph.nodes[neighbor_job_id]["duration"]

    # Build solution up to job k
    new_solution = {}
    time_step_to_job_id = {}
    for job_id in range(1, len(solution.keys())+1):
        if (job_starting_time:=solution[job_id]) <= original_neighbor_starting_time and job_id != neighbor_job_id:
            new_solution[job_id] = job_starting_time
            if job_starting_time in time_step_to_job_id.keys():
                time_step_to_job_id[job_starting_time].append(job_id)
            else:
                time_step_to_job_id[job_starting_time] = [job_id]

    # Init search params
    current_time = 0
    available_ressources = np.array(r.resource_availabilities)
    available_ressources_through_time = [copy(available_ressources) for _ in range(original_neighbor_starting_time)]
    jobs_done_through_time = [set() for _ in range(original_neighbor_starting_time+1)]


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

    current_time = 0
    # Step #2: Try to insert the neighbor_job_id before it's inital starting time without moving the other node
    while current_time <= original_neighbor_starting_time:
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
    new_solution = randomdly_insert_missing_job(r, new_solution)

    return remove_delay(r, new_solution)


def remove_delay(r: RCPSP, solution: Dict[int, int]):

    # Find which task is starting at each timestep (reverse the dictionnary)
    time_to_job = {}
    for job in solution.keys():
        if solution[job] in time_to_job:
            if r.graph.nodes[job]['duration'] == 0:
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
                available_ressources_through_time = find_ressources_used_at_each_timestep(r, optimised_solution, current_time)
                job_ressources = r.graph.nodes[job]['resources']
                while insert_time >= 0:

                    # Check the precedence constraint
                    constraint_violated = False
                    precendent_jobs = r.graph.predecessors(job)
                    for precendent_job in precendent_jobs:
                        duration = r.graph.nodes[precendent_job]['duration']
                        constraint_violated |= optimised_solution[precendent_job] + duration > insert_time # Todo: Verif if's right
                    if constraint_violated:
                        break

                    # Check ressources availability
                    if min(available_ressources_through_time[insert_time] - job_ressources) >= 0:
                        earliest_insert_time_found = insert_time
                    else:
                        break

                    insert_time -= 1
                
                optimised_solution[job] = earliest_insert_time_found

        current_time += 1     

    return optimised_solution


def optimise_by_swap(r: RCPSP, solution: Dict[int, int]):
    """
    1. Find the job where the delay between the end and the begening of its nearest sucessors is the longest.
    2. Find which job can be swap with the job identified in step 1.
    3. Swap the job and rebuild the solution
    """
    # Step 1.
    job_to_swap, longuest_downtime = None, -1

    for job in solution.keys():
        shortest_downtime = inf
        job_duration = r.graph.nodes[job]['duration']
        for job_successor in r.graph.successors(job):
            start = solution[job]
            start_new = solution[job_successor]
            shortest_downtime = min(shortest_downtime, solution[job_successor] - (solution[job] + job_duration))
        if shortest_downtime > longuest_downtime and shortest_downtime != inf:
            longuest_downtime = shortest_downtime
            job_to_swap = job

    if longuest_downtime == 0:
        return solution

    job_execution_order = sorted([(job, solution[job]) for job in solution.keys()], key=lambda x: x[1])
    job_execution_order = [job for job, starting_time in job_execution_order]
    job_to_swap_idx = job_execution_order.index(job_to_swap)

    if job_to_swap_idx+1 >= len(job_execution_order)-1: # -1 since last job is "useless" (not swapable)
        return solution
    
    # Step 2.
    current_time = solution[job_to_swap]
    new_solution = {job: solution[job] for job in solution.keys() if solution[job] <= current_time}
    # last_job_time = solution[job_to_swap] if solution[job_execution_order[job_to_swap_idx+1]] == solution[job_to_swap] else solution[job_execution_order[job_to_swap_idx-1]]
    # ressource_available = np.array(find_ressources_used_at_each_timestep(r, solution, last_job_time)[-1])
    last_time_step = solution[job_execution_order[-1]]

    # Find which jobs can be swap with the current one
    swapable_jobs_dics = {}
    while current_time < last_time_step:
        
        ressource_available = np.array(find_ressources_used_at_each_timestep(r, new_solution, current_time)[-1])
        ptr = job_to_swap_idx + 1

        while ptr < r.graph.number_of_nodes():
            if solution[job_execution_order[ptr]] == solution[job_to_swap]:
                ptr += 1
                continue
            potential_job_swap = job_execution_order[ptr]
            ressource_needed = np.array(r.graph.nodes[potential_job_swap]['resources'])
            precedence_constraint_violated = is_precedence_constraint_violated(r, new_solution, potential_job_swap, current_time)
            already_in_solution = potential_job_swap in new_solution
            if min(ressource_available - ressource_needed) >= 0 and not precedence_constraint_violated and not already_in_solution:
                # Swap is possible
                swapable_jobs_dics[potential_job_swap] = current_time
            ptr += 1
        current_time += 1

        if current_time > solution[job_to_swap] + r.graph.nodes[job_to_swap]['duration']:
            break

    
    # Step 3. Swap the job and rebuild the solution
    current_time = solution[job_to_swap]

    # Step 3.1 Add the job that are swapable
    # To randomly choose a job to swap according to their importance, we need to define their importance
    def importance_probability(list_of_job):
        probabilities = np.array([len(list(r.graph.successors(job))) + 1 for job in list_of_job])
        return probabilities / np.sum(probabilities)

    swapable_jobs = [job for job in swapable_jobs_dics.keys()]
    for _ in range(len(swapable_jobs_dics.keys())//2):
        job_to_insert = np.random.choice(swapable_jobs, p=importance_probability(swapable_jobs))
        insert_time = swapable_jobs_dics[job_to_insert]
        ressource_needed = np.array(r.graph.nodes[job_to_insert]['resources'])
        
        can_insert = True
        for time in range(insert_time, insert_time + r.graph.nodes[job_to_insert]['duration']):
            ressource_available = find_ressources_used_at_each_timestep(r, new_solution, insert_time + 1)[-1]  
            if np.min(ressource_available - ressource_needed) < 0:
                can_insert = False
        if can_insert:
            new_solution[job_to_insert] = insert_time

        del swapable_jobs[swapable_jobs.index(job_to_insert)]

    # Step 3.2 Add the remaining job
    new_solution = randomdly_insert_missing_job(r, new_solution)

    return remove_delay(r, new_solution)
        