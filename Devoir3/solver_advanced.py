from typing import List, Tuple
from rcpsp import RCPSP
from time import time, sleep
from tqdm import tqdm
import random
import numpy as np

def solve(rcpsp: RCPSP) -> List[int]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        rcpsp (RCPSP): object containing the graph for the instance to solve

    Returns:
        solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation 
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot
    """
    # Add here your solving process here

    print(f"Ressource availabilities: {rcpsp.resource_availabilities}\n")
    print(f"Nodes {len(rcpsp.graph.nodes)}: {rcpsp.graph.nodes}\n")
    print(f"Nodes {rcpsp.graph.nodes[10]}\n")
    print(f"Edges {len(rcpsp.graph.edges)}: {rcpsp.graph.edges}\n")


    return algo_name(rcpsp, 30)


def algo_name(r: RCPSP, time_limit):

    start_time = time()
    tic = start_time

    initial_solution = generate_solution(r)

    # with tqdm(total=time_limit) as progress_bar:
    #     while (tac:=time()) - start_time < time_limit:
    #         progress_bar.update(tac - tic)
    #         tic = tac
            
    #         break
            
    #         sleep(2)

    print(initial_solution)
    return initial_solution


def generate_solution(r: RCPSP):
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
    
        for _ in range(len(available_job)):
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
