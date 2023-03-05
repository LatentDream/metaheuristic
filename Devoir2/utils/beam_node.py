from copy import deepcopy


class BeamNode:

    def __init__(self, node_id, pheromone):
        self.id = node_id
        self.pheromone = deepcopy(pheromone)
        self.children = set()
