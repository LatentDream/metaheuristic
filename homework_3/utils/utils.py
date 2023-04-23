from copy import copy
from typing import List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np



def generate_random_valid_solution(r: RCPSP):
    """Generate a random feasible solution for the RCPSP problem using earliest start time algorithm"""
    job_id = 1
    current_time = 0
    solution = {job_id: current_time}
    available_job = list(r.graph.successors(job_id))
    available_ressources = np.array(r.resource_availabilities)
    locked_ressources = dict()
    next_job = dict()

    while len(available_job) != 0 or len(next_job.keys()) != 0:

        # Free newly available ressource
        if current_time in locked_ressources.keys():
            available_ressources += locked_ressources[current_time]
            del locked_ressources[current_time]
        # Add new job that can be done
        if current_time in next_job.keys():
            available_job += next_job[current_time]
            del next_job[current_time]

        # Try to lunch a random job
        if len(available_job) == 0:
            current_time += 1
            continue
    
        for _ in range(max(int(len(available_job)/2), 1)):
            job_idx = random.randrange(0, len(available_job))
            job_id = available_job[job_idx]
            ressources_needed_to_start_job = np.array(r.graph.nodes[job_id]["resources"])

            # Check the precedence constraints
            prcedent_jobs = list(r.graph.predecessors(job_id))
            constraints_violated = False
            for job in prcedent_jobs:
                if job in solution.keys() and not job in next_job.keys():
                    constraints_violated &= False
                else:
                    constraints_violated &= True
            if constraints_violated:
                continue

            # Launch the job
            if min(available_ressources - ressources_needed_to_start_job) >= 0:
                solution[job_id] = current_time
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

    
    return solution



def find_ressources_used_at_each_timestep(r: RCPSP, solution, up_to_time: int=None):

    # Find which task is starting at each timestep (reverse the dictionnary)
    time_to_job = {}
    max_time = -1
    for job in solution.keys():
        if solution[job] in time_to_job:
            time_to_job[solution[job]].append(job)
        else:
            time_to_job[solution[job]] = [job]
        if t:=(solution[job]) > max_time:
            max_time = t
    
    if up_to_time==None:
        up_to_time = max_time

    current_time = 0
    available_ressources = np.array(r.resource_availabilities)
    available_ressources_through_time = [copy(available_ressources) for _ in range(up_to_time)]

    while current_time <= up_to_time:
        if current_time in time_to_job:
            for curr_job_id in time_to_job[current_time]:
                curr_job_ressource = np.array(r.graph.nodes[curr_job_id]["resources"])
                duration = r.graph.nodes[curr_job_id]["duration"]
                locked_until = current_time + duration
                # remove the ressources for the curr_job
                for i in range(current_time, min(up_to_time, locked_until)):
                    available_ressources_through_time[i] -= curr_job_ressource
        current_time += 1
    

    return available_ressources_through_time


def randomdly_insert_missing_job(r: RCPSP, solution):
    current_time = 0
    available_job = [1]
    available_ressources = np.array(r.resource_availabilities)
    locked_ressources = dict()
    next_job = dict()

    time_solution_is_build_up_to = max([solution[job] for job in solution.keys()])

    time_step_to_job_id = {}
    for job_id in sorted(solution.keys()):
        job_starting_time = solution[job_id]
        if job_starting_time in time_step_to_job_id.keys():
            time_step_to_job_id[job_starting_time].append(job_id)
        else:
            time_step_to_job_id[job_starting_time] = [job_id]


    # Do all the jobs that are already in new_solution
    while current_time <= time_solution_is_build_up_to:

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
                    else:
                        next_job[locked_until] = next_available_job
                else:
                    available_job += next_available_job
                del available_job[available_job.index(job_id)]
        current_time += 1


    # Do the other job
    while len(solution.keys()) != r.graph.number_of_nodes():

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
                if job_id in solution.keys():
                    del available_job[job_idx]
                    continue
                ressources_needed_to_start_job = np.array(r.graph.nodes[job_id]["resources"])

                # Check the precedence constraints
                precedent_jobs = set(r.graph.predecessors(job_id))        
                constraints_violated = not all([job in solution.keys() and solution[job] + r.graph.nodes[job]["duration"] < current_time for job in precedent_jobs])
                if constraints_violated:
                    continue

                # Launch the job
                if min(available_ressources - ressources_needed_to_start_job) >= 0:
                    solution[job_id] = current_time
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

    return solution