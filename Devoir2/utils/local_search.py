

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

def swap(solution: List[int], k: int, tsptw: TSPTW) -> List[int]:
    """
    # Algo 4 to speed up calcul of f and delta
    P : tour
    k : position in the tour
    c(a_i): cost from the travel
    tsptw.time_window : [e_i, l_i]
    D_p_k = max(A_p_k, e_p_k)
    A_p_k = D_p_k + c(a_p_k-1,p_k): arrival time at customer p_k
    
    minimize f(P) = sum(c)
    subject to delta(P) = sum(w(p_k)) = 0

    w(p_k) = 
    """
    #! This is overkill for now, start by testing the do_swap and recalcul the hole thing after
    f_old = tsptw.get_solution_cost(solution)
    delta_old = get_number_of_violations(solution, tsptw)
    cost = delta_c(solution, k, tsptw)

    # if A_p_k > l_p_k then delta_old := delta_old - 1
    A_p_k = get_solution_cost_up_to_k(solution, k, tsptw)
    if A_p_k > tsptw.time_windows[k][0]:
        delta_old -= 1
    # if A_p_k+1 > l_p_k+1 then delta_old := delta_old - 1
    A_p_k_1 = get_solution_cost_up_to_k(solution, k+1, tsptw)
    if A_p_k_1 > tsptw.time_windows[k+1][0]:
        delta_old -= 1
    # if A_p_k+2 > l_p_k+2 then delta_old := delta_old - 1
    A_p_k_2 = get_solution_cost_up_to_k(solution, k+2, tsptw)
    if A_p_k_2 > tsptw.time_windows[k+2][0]:
        delta_old -= 1

    # A_p_k = max(A_p_k-1 + c(p_k-1, p_k+1), e_p_k+1)
    A_p_k = max(get_solution_cost_up_to_k(solution, k-1, tsptw) + tsptw.graph[k-1][k+1]["weight"], tsptw.time_windows[k+1][0])
    # A_p_k+1 = max(A_p_k + c(p_k+1, p_k-1), e_p_k)
    A_p_k_1 = max(get_solution_cost_up_to_k(solution, k, tsptw) + tsptw.graph[k+1][k-1]["weight"], tsptw.time_windows[k][0])
    # A_p_k+2 = max(A_p_k+1 + c(p_k, p_k+2), e_p_k+2)
    A_p_k_2 = max(get_solution_cost_up_to_k(solution, k+1, tsptw) + tsptw.graph[k][k+2]["weight"], tsptw.time_windows[k+2][0])


    # if A_p_k > l_p_k+1 then delta_old := delta_old - 1
    if A_p_k > tsptw.time_windows[k+1][0]: delta_old -= 1
    # if A_p_k+1 > l_p_k then delta_old := delta_old - 1
    if A_p_k_1 > tsptw.time_windows[k][0]: delta_old -= 1
    # if A_p_k > l_p_k+2 then delta_old := delta_old - 1
    if A_p_k > tsptw.time_windows[k+2][0]: delta_old -= 1

    raise Exception(f"{swap.__name__} is not implemented")


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
