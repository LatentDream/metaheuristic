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

    def __init__(self, tsptw: TSPTW, l_rate:float=0.1, tau_max:float=0.999, tau_min:float=0.001):
        self.n = tsptw.num_nodes
        self.pheromone = [[0.5] * self.n] * self.n
        self.uniforme = np.random.uniform
        self.l_rate = l_rate
        self.tau_max = tau_max
        self.tau_min = tau_min


    def resetUniformPheromoneValues(self):
        self.pheromone = [[0.5] * self.n] * self.n


    def updatePheromoneValues(self, bs_update, cf, p_ib, p_rb, p_bf):
        """
        Three solutions are used for updating the pheromone values. These are the iteration-best
        solution Pib, the restart-best solution Prb, and the best-so-far solution Pbf. The influence 
        of each solution on the pheromone update depends on the state of convergence of the algorithm 
        as measured by the convergence factor cf
            tau_ij = tau_if + l_rate * (eps_ij - tau_ij)
          where
            eps_ij = k_ib * p_ib_ij + k_rb * p_rb_ij + k_bf * p_bf_ij
            p_*_ij is 1 if customer j is visited after customer i in solution P and 0 otherwise
        """
        k_ib, k_rb, k_bf = self.__get_ks(bs_update, cf)
        
        for i in range(self.n):
            # find customer i #! 0 to n-1 -> add one #TODO Verif if it's right
            p_ib_idx, p_rb_idx, p_bf_idx = p_ib.index(i+1), p_rb.index(i+1), p_bf.index(i+1)
            for j in range(self.n):
                # eps_ij = k_ib * p_ib_ij + k_rb * p_rb_ij + k_bf * p_bf_ij
                eps_ij = k_ib * float(p_ib[p_ib_idx+1]==(j+1)) + k_rb * float(p_rb[p_rb_idx+1]==(j+1)) + k_bf * float(p_bf[p_bf_idx+1]==(j+1))
                # tau_ij = tau_if + l_rate * (eps_ij - tau_ij)
                self.pheromone[i][j] += self.l_rate * (eps_ij - self.pheromone[i][j])
                # Avoid complete convergence
                self.pheromone[i][j] = min(max(self.pheromone[i][j], self.tau_min), self.tau_max)


    def __get_ks(self, bs_update, cf):
        """ Return k_ib, k_rb, k_bf (Value from paper table #1) """
        if bs_update:
            return 0.0, 0.0, 1.0
        if cf < 0.4:
            return 1.0, 0.0, 0.0
        if cf < 0.6:
            return 2/3, 1/3, 0
        if cf <= 1.0:
            return 0.0, 1.0, 0.0
        raise ValueError("cf:({cf}) is greater than 1")


    def stochastic_sampling(self, n_samples, determininsm_rate):
        raise Exception(f"{self.stochastic_sampling.__name__} is not implemented")


    def construction(self, determinism_rate):
        raise Exception(f"{self.construction.__name__} is not implemented")


    def beam_construct(self, determinism_rate, beam_with, max_children, to_choose, n_samples, sample_rate):
        raise Exception(f"{self.beam_construct.__name__} is not implemented")



