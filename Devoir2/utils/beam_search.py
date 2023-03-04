from typing import List
from tsptw import TSPTW
import numpy as np


class ProbabilisticBeamSearch():

    def __init__(self, determinism_rate, beam_width, max_children, to_choose, n_samples, sample_rate):
        self.determinism_rate = determinism_rate
        self.beam_width = beam_width
        self.max_children = max_children
        self.to_choose = to_choose
        self.n_samples  = n_samples 
        self.sample_rate = sample_rate


    def beam_construct(self):
        return self.__beam_construct() if self.beam_width > 1 else self.__construction()
        

    def __construction(self):
        raise Exception(f"{self.construction.__name__} is not implemented")


    def __beam_construct(self):
        """
        """
        last_customer_added = 0



        raise Exception(f"{self.beam_construct.__name__} is not implemented")
      
    
    def __stochastic_sampling(self, pheromone: List[List[float]], last_customer_added):
        """
        At each step, the set of unvisited costumers is denoted by N(P). Once all customers
        have been added to the tour, it is completed by adding node 0 which represents the depot. The
        decision of which customer to choose at each step is done with the help of pheromone information and
        heuristic information.
        """
        # first generate a random number q uniformly distributed [0; 1] 
        q = np.random.uniform(0.0, 1.0)
        # Compare this value with the determinism rate
        tau_eta = np.array(pheromone[last_customer_added]) *  np.array(self.__heuristic_benefit_of_visiting_customer())
        if q <= self.determinism_rate:
            # j in N(P) is choosen deterministically as costumer with highest product of pheromone and heuristic
            return np.argmax(tau_eta)
        else:
            # j stochastically chosen from the following distribtion
            return np.random.choice([i for i in range(len(tau_eta))], p=tau_eta / sum(tau_eta))
    

    def __heuristic_benefit_of_visiting_customer(self):
        """
        Regarding the definition of n_ij, several existing greedy functions for the TSPTW may be used for that purpose. When deciding which customer should be visited next, not only a small travel cost between customers is desirable but also those customers whose time window finishes sooner should be given priority to avoid constraint violations. In addition, visiting those customers whose time window starts earlier may prevent
        waiting times. Hence
        """
        # TODO: implement an heuristic
        # lambda_c = np.random.uniform(0, 1.0)
        # lambda_l = np.random.uniform(0, 1.0-lambda_c)
        # lambda_e = 1.0 - lambda_c - lambda_l

        return np.array([1.0]*self.n)
    



