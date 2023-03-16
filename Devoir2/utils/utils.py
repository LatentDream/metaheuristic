from copy import deepcopy
from typing import List

from tsptw import TSPTW


def get_number_of_violations(solution: List[int], tsptw: TSPTW) -> int:
    nb_of_violation = 0
    time_step = 0
    last_stop = 0
    for next_stop in solution[1:]:
        edge = (last_stop, next_stop)
        time_step += tsptw.graph.edges[edge]["weight"]
        time_windows_begening, time_windows_end = tsptw.time_windows[next_stop]
        if  time_step < time_windows_begening:
            waiting_time = time_windows_begening - time_step
            time_step += waiting_time
        if time_step > time_windows_end:
            nb_of_violation += 1
    
    return nb_of_violation

def get_best_soltion(solution1, solution2, tsptw) -> List[int]:
    if solution1 == None and solution2 == None:
        return None
    if solution1 == None:
        return deepcopy(solution2)
    if solution2 == None:
        return deepcopy(solution1)
    if get_number_of_violations(solution1, tsptw) == 0 and get_number_of_violations(solution2, tsptw) > 0:
        return deepcopy(solution1)
    if get_number_of_violations(solution1, tsptw) > 0 and get_number_of_violations(solution2, tsptw) == 0:
        return deepcopy(solution2)
    return deepcopy(solution1) if get_score(solution1, tsptw) < get_score(solution2, tsptw) else deepcopy(solution2)


def get_score(solution: List[int], tsptw: TSPTW) -> int:   
    # ? Change for an estimation ? 
    return tsptw.get_solution_cost(solution)
