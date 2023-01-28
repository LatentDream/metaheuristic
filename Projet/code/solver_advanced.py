import numpy as np
import copy
import time
import random


def solve_advanced(eternity_puzzle):
    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    
    best_solution = generate_random_solution(eternity_puzzle)
    best_solution_cost = eternity_puzzle.get_total_n_conflict(best_solution)
    best_candidate= best_solution
    
    tabuList = [best_solution]
    maxTabuListLength = 100
    start_time = time.time()
    elapsed_time = 0
    search_time = 60*60
    
    
    while elapsed_time < search_time:

        neighborhood = getNeighbors(best_candidate,eternity_puzzle)
        
        best_candidate = neighborhood[0]
        best_candidate_cost = eternity_puzzle.get_total_n_conflict(best_candidate)
        
        for candidate in neighborhood:
            a = eternity_puzzle.get_total_n_conflict(candidate)
            if ((not candidate in tabuList) and (a < best_candidate_cost)):
                best_candidate = candidate
                best_candidate_cost = a
                
        if best_candidate_cost < best_solution_cost:
            best_solution = best_candidate
            best_solution_cost = best_candidate_cost
            
        tabuList.append(best_candidate)
        
        if len(tabuList) > maxTabuListLength:
            tabuList = tabuList[1:]
        
        elapsed_time = time.time() - start_time
        
        if best_solution_cost == 0:
            return best_solution,best_solution_cost
    
    return best_solution, best_solution_cost
        
    

def generate_random_solution(eternity_puzzle):

    solution = []
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    for i in range(eternity_puzzle.n_piece):
        range_remaining = np.arange(len(remaining_piece))
        piece_idx = np.random.choice(range_remaining)
        piece = remaining_piece[piece_idx]
        permutation_idx = np.random.choice(np.arange(4))
        piece_permuted = eternity_puzzle.generate_rotation(piece)[permutation_idx]
        solution.append(piece_permuted)
        remaining_piece.remove(piece)

    return solution

# 2 swap with rotations neighbourhood 
def getNeighbors(solution,eternity_puzzle):
    neighbourhood = []
        
    for i in range(len(solution)):
        neighbor1 = solution.copy()

        for rotated_piece in eternity_puzzle.generate_rotation(neighbor1[i]):
            if rotated_piece != neighbor1[i]:
                neighbor1[i] = rotated_piece
                neighbourhood.append(neighbor1)
        
        for j in range(len(solution)):

            if i != j:
                neighbor2 = solution.copy()
                neighbor2[i], neighbor2[j] = neighbor2[j], neighbor2[i]
                
                for rotated_piece1 in eternity_puzzle.generate_rotation(neighbor2[i]):
                    for rotated_piece2 in eternity_puzzle.generate_rotation(neighbor2[j]):
                        neighbor2[i] = rotated_piece1
                        neighbor2[j] = rotated_piece2
                        neighbourhood.append(neighbor2)  
                        
    return neighbourhood



