import numpy as np
import math
import random
import time


def local_search(magic_square, neighbour_function, validity_function, eval_function, seed):

    start_time = time.time()

    solution = generate_random_solution(magic_square, seed)
    neighboorhood = neighbour_function(solution)
    valid_neighboorhood = validity_function(magic_square, neighboorhood, solution, eval_function)

    i = 0

    while len(valid_neighboorhood) > 0:

        print("STEP", i)
        print("number of conflicts:", eval_n_conflict(magic_square, solution))
        print("degree of conflicts:", eval_degree_conflict(magic_square, solution))
        print(solution)
        print("-----")

        solution = min(valid_neighboorhood, key=lambda x: eval_function(magic_square, x))
        
        neighboorhood = neighbour_function(solution)
        valid_neighboorhood = validity_function(magic_square, neighboorhood, solution, eval_function)
        i += 1
        
    end_time = time.time() - start_time
        
        
    print("SOLUTION OBTAINED")
    print("average time per iteration:", round(end_time / i, 2), "seconds")
    print("number of conflicts:", eval_n_conflict(magic_square, solution))
    print("degree of conflicts:", eval_degree_conflict(magic_square, solution))
    print(solution)
    print("-----")
    return solution

# Initialization functions #
def generate_random_solution(magic_square, seed):
    """
    Generate a random solution (permutation of numbers) for a 
    magic square of size n*n.
    """
    solution = np.arange(1, magic_square.size * magic_square.size + 1)
    np.random.seed(seed)
    np.random.shuffle(solution)
    solution = solution.reshape(magic_square.size, magic_square.size)
    return solution
        
# Neighboorhood function #
def generate_two_swap_neighboorhood(solution):
    """
    Neighboorhood function for the 2-swap
    Return the neighbours of the current solution obtained by two-swap move
    """
        
    neighbourhood = []
        
    for i1 in range(solution.shape[0]):
        for j1 in range(solution.shape[0]):
            for i2 in range(solution.shape[0]):
                for j2 in range(solution.shape[0]):
                        
                    neighbour = swap(solution, i1, j1, i2, j2)
                    neighbourhood.append(neighbour)
    return neighbourhood

def generate_three_swap_neighboorhood(solution):
    """
    Neighboorhood function for the 3-swap
    Return the neighbours of the current solution obtained by two-swap move
    WARNING: this function is not optimized at all. It is much more efficient to remove directly the 
    invalid neighboors in the list rather than filter the whole list afterwards
    """
    neighbourhood = []
    
    for i1 in range(solution.shape[0]):
        for j1 in range(solution.shape[0]):
            for i2 in range(solution.shape[0]):
                for j2 in range(solution.shape[0]):
                    for i3 in range(solution.shape[0]):
                        for j3 in range(solution.shape[0]):
                    
                            neighbour = swap(solution, i2, j2, i3, j3)
                            neighbourhood.append(neighbour)
                                
                            neighbour = swap(solution, i1, j1, i2, j2)
                            neighbourhood.append(neighbour)    
                                
                            neighbour = swap(solution, i1, j1, i3, j3)
                            neighbourhood.append(neighbour)
                                
                            neighbour = swap(solution, i1, j1, i2, j2)
                            neighbour = swap(neighbour, i2, j2, i3, j3)
                            neighbourhood.append(neighbour)  
                                
                            neighbour = swap(solution, i2, j2, i3, j3)
                            neighbour = swap(neighbour, i1, j1, i2, j2)
                            neighbourhood.append(neighbour)
        
    return neighbourhood

def swap(solution, i1, j1, i2, j2):
    """
    return a new solution where the value of two cells are swapped
    """
    swapped_array = solution.copy()
    tmp = swapped_array[i1,j1]
    swapped_array[i1,j1] = swapped_array[i2,j2]
    swapped_array[i2,j2] = tmp
    return swapped_array

# Validity function #
def is_improving_validity_function(magic_square, neighboorhood, solution, eval_function):
    """
    Return only the list of neighbours that are improving the evaluation cost
    """
    
    return [n for n in neighboorhood if eval_function(magic_square, n) < eval_function(magic_square, solution)]

def accept_all_neighboors(neighboorhood):
    """
    All the neighboors are valid (used in simulated annealing)
    """
    
    return neighboorhood

# Evaluation functions #
def eval_n_conflict(magic_square, solution):
    """
    Evaluation function for magic square: number of conflicts
    """

    column_sum = np.sum(solution, axis=0)
    column_conflict = sum([x != magic_square.total for x in column_sum])
        
    row_sum = np.sum(solution, axis=1)
    row_conflict = sum([x != magic_square.total for x in row_sum])
        
    low_diag_conflict = np.trace(solution) != magic_square.total
    up_diag_conflict =  np.trace(np.fliplr(solution)) != magic_square.total
        
    return column_conflict + row_conflict + low_diag_conflict + up_diag_conflict 

def eval_degree_conflict(magic_square, solution):
    """
    Evaluation function for magic square: weighted number of conflicts
    """
        
    column_sum = np.sum(solution, axis=0)
        
    column_conflict = sum([abs(x - magic_square.total) for x in column_sum])
        
    row_sum = np.sum(solution, axis=1)
    row_conflict = sum([abs(x - magic_square.total) for x in row_sum])
        
    low_diag_conflict = abs(np.trace(solution) - magic_square.total)
    up_diag_conflict =  abs(np.trace(np.fliplr(solution)) - magic_square.total)
    return column_conflict + row_conflict + low_diag_conflict + up_diag_conflict 
