from copy import deepcopy
from typing import List
import numpy as np

from tsptw import TSPTW

class BeamNode:

    def __init__(self, node_id, pheromone):
        self.id = node_id
        self.pheromone = deepcopy(pheromone)
        self.pheromone[:, self.id] = 0. # C := C\<P,j>
        self.children = dict()


    def extract_solution(self) -> List[List[int]]:
        def recursive_solution_builder(beam_node: BeamNode, solution: List[int], solutions_found: List[List[int]]):
            solution.append(beam_node.id)
            if len(beam_node.children.keys()) == 0:
                solutions_found.append(solution)
            else:
                for child_id in beam_node.children.keys():
                    recursive_solution_builder(beam_node.children[child_id], deepcopy(solution), solutions_found)
                          
        solutions_found = []
        recursive_solution_builder(self, [], solutions_found)
        return solutions_found

    
    def extract_best_solution(self, tsptw: TSPTW):
        potential_solutions = self.extract_solution()
        potential_solutions_cost = [tsptw.get_solution_cost(solution) for solution in potential_solutions]
        return potential_solutions[np.argmin(potential_solutions_cost)]  
        

    def create_child(self, child_id: int):
        child = BeamNode(child_id, self.pheromone)
        return child

    def add_children(self, child):
        self.children[child.id] = child
        return child

    def create_and_add_child(self, child_id: int): 
        return self.add_children(self.create_child(child_id))
        