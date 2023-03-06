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
        self.pheromone = np.array([np.array([0.5] * self.n)] * self.n)
        self.uniforme = np.random.uniform
        self.l_rate = l_rate
        self.tau_max = tau_max
        self.tau_min = tau_min


    def resetUniformPheromoneValues(self):
        self.pheromone = np.array([np.array([0.5] * self.n)] * self.n)


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
            # find customer i
            p_ib_idx, p_bf_idx = p_ib.index(i), p_bf.index(i)
            p_rb_idx = p_rb.index(i) if p_rb != None else -1
            for j in range(self.n):
                # eps_ij = k_ib * p_ib_ij + k_rb * p_rb_ij + k_bf * p_bf_ij
                eps_ib_ij = k_ib * float(p_ib[p_ib_idx+1]==(j)) 
                eps_bf_ij = k_bf * float(p_bf[p_bf_idx+1]==(j))
                eps_rb_ij = k_rb * float(p_rb[p_rb_idx+1]==(j)) if p_rb != None else 0.
                eps_ij = eps_ib_ij + eps_rb_ij + eps_bf_ij
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
    
    
    def computeConvergenceFactor(self):
        # Equation #6
        convergence_factor = 0.0
        for i in range(self.n):
            for j in range(self.n):
                convergence_factor += max(self.tau_max - self.pheromone[i][j], self.pheromone[i][j] - self.tau_min)
        convergence_factor = convergence_factor / self.n*self.n * (self.tau_max - self.tau_min)
        convergence_factor = 2.0 * (convergence_factor -0.5)
        return convergence_factor
