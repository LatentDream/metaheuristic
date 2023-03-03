from tsptw import TSPTW
import numpy as np



class Ant: 
    """
    Modeled on the actions of an ant colony.[4] Artificial 'ants' (e.g. simulation agents) 
    locate optimal solutions by moving through a parameter space representing all possible 
    solutions. Real ants lay down pheromones directing each other to resources while exploring 
    their environment. The simulated 'ants' similarly record their positions and the quality 
    of their solutions, so that in later simulation iterations more ants locate better solutions.
    """

    def __init__(self, tsptw: TSPTW):
        self._n = tsptw.num_nodes
        self.pheromone = [[0.5] * self._n] * self._n
        self.uniforme = np.random.uniform

    def resetUniformPheromoneValues(self):
        self.pheromone = [[0.5] * self._n] * self._n

    def stochastic_sampling(n_samples, determininsm_rate):
        pass

    def construction(determinism_rate):
        pass

    def beam_construct(determinism_rate, beam_with, max_children, to_choose, n_samples, sample_rate):
        # 
        pass


