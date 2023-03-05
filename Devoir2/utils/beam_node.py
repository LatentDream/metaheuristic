from copy import deepcopy
from typing import List


class BeamNode:

    def __init__(self, node_id, pheromone):
        self.id = node_id
        self.pheromone = deepcopy(pheromone)
        self.children = set()



    def extract_solution(self):
        def recursive_solution_builder(beam_node: BeamNode, solution: List[int], solutions_found: List[List[int]]):
            solution.append(beam_node.id)
            if len(beam_node.children) == 0:
                solutions_found.append(solution)
            else:
                for child in beam_node.children:
                    recursive_solution_builder(child, deepcopy(solution), solutions_found)
                          
        solutions_found = []
        recursive_solution_builder(self, [], solutions_found)
        return solutions_found

