

from copy import deepcopy
from typing import List
from tsptw import TSPTW
from utils.utils import get_best_soltion, get_number_of_violations


def local_search(solution: List[int], tsptw: TSPTW) -> List[int]:
    #?: This part is the algo #3 of the paper
    # based on the 1-opt neighborhood in which a single customer is 
    # removed from the tour and reinserted in a different position

    # Remove start and end to warehouse
    # warehouse_id = solution[0]
    # assert warehouse_id == solution[-1]
    # del solution[0]
    # del solution[-1]

    p_best = deepcopy(solution)
    #! for k in range(tsptw.num_nodes-3):
    for k in range(1, len(solution)-2):
        p_test = deepcopy(solution)
        if not is_time_window_infeasible(p_test[k], p_test[k+1], p_test, tsptw):
            p_test = do_swap(p_test, k) # Algo #4
            p_best = get_best_soltion(p_test, p_best, tsptw)
            p_test2 = deepcopy(p_test)
            #! for d in range(k+1, tsptw.num_nodes-3):
            for d in range(k+1, len(solution)-2):
                if is_time_window_infeasible(p_test[d], p_test[d+1], p_test, tsptw):
                    break
                p_test = do_swap(p_test, d)
                p_best = get_best_soltion(p_test, p_best, tsptw)
            p_test = p_test2
            #! for d in range(k-1, 0):
            for d in range(k-1, 1):
                if is_time_window_infeasible(p_test[d], p_test[d+1], p_test, tsptw):
                    break
                p_test = do_swap(p_test, d)
                p_best = get_best_soltion(p_test, p_best, tsptw)

    # Put back to start and end to warehouse
    # solution.insert(0, warehouse_id)
    # solution.append(warehouse_id)

    return p_best


def is_time_window_infeasible(last_stop: int, next_stop: int, solution: List[int], tsptw: TSPTW):
    # Find current time from the solution
    current_time = 0
    for k in range(tsptw.num_nodes - 1):
        if solution[k] == last_stop:
            break
        current_time += tsptw.graph.edges[(solution[k], solution[k+1])]["weight"]
        if current_time < (start_window:=tsptw.time_windows[solution[k+1]][0]):
            current_time += start_window - current_time      
    # Add travel time
    current_time = tsptw.graph.edges[(last_stop, next_stop)]["weight"]
    # Check if infeasible
    if current_time > (end_time_window_client:=tsptw.time_windows[solution[k+1]][1]):
        return True
    return False   


def do_swap(solution: List[int], k: int):
    """ Exchange the customers at positions k and k+1. """
    solution[k], solution[k+1] = solution[k+1], solution[k]
    return solution


def delta_c(solution: List[int], k: int, tsptw: TSPTW):
    cost = sum([tsptw.graph.edges[(solution[idx], solution[idx+1])]["weight"] for idx in range(k, tsptw.num_nodes-1)])
    cost -= sum([tsptw.graph.edges[(solution[idx], solution[idx+1])]["weight"] for idx in range(k, 0)])
    return cost


def get_solution_cost_up_to_k(solution: List[int], k: int, tsptw: TSPTW) -> int:
    total_time = 0
    for i in range(k-1):
        current_node = solution[i]
        next_node = solution[i+1]
        lower_bound, upper_bound = tsptw.time_windows[next_node]
        total_time += tsptw.graph[current_node][next_node]["weight"]
        total_time = max(total_time, lower_bound)
    
    return total_time
