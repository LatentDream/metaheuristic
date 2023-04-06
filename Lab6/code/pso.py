import sys
import numpy as np
import time as t

class PSO:

    def __init__(self, particles, velocities, fitness_function, w=0.8, cog=1, soc=1, max_iter=10, max_time=10, auto_coef=True):
        """Initialisation of PSO solving. A solution is a feasible vector that allows to compute a cost using the fitness function
        :param particles: the initial array representing the solutions
        :param velocities: the initial array representing the velocities of the particles
        :fitness_function: the objective function taking an array of particle in parameter and returning the array of their costs

        :param w: inertia coefficient
        :param cog: cognitive coefficient
        :param soc: social coefficient
        :param max_iter: maximum number of particles update
        :param max_time: maximum computation time
        :param auto_coef: if True, the parameters are updated at each iteration
        """

        # Particles data
        self.particles = particles
        self.size = particles.shape[1]
        self.N = len(self.particles)
        self.velocities = velocities
        self.fitness_function = fitness_function
        
        # ICS weights (Inertia, Cognitive, Social)
        self.w = w
        self.cog = cog
        self.soc = soc
        self.auto_coef = auto_coef

        # Best known variables
        self.p_bests = self.particles
        self.p_bests_values = self.fitness_function(self.particles)
        self.g_best = self.p_bests[0]
        self.g_best_value = self.p_bests_values[0]
        self.update_bests()

        # End criterion
        self.max_iter = max_iter
        self.max_time = max_time
        self.start_time = t.time()
        self.iter = 0
        self.is_running = True
        self.update_coef()

    def update_coef(self):
        """Updates ICS coefficients according to Clerc and Kennedy formulas"""

        if self.auto_coef:
            t = self.iter
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.cog = -3 * t / n + 3.5
            self.soc =  3 * t / n + 0.5

    def update_bests(self):
        """Updates best known solutions (personal and global)"""

        costs = self.fitness_function(self.particles)

        for i in range(len(self.particles)):

            # Updating best personnal value 
            if costs[i] < self.p_bests_values[i]:
                self.p_bests_values[i] = costs[i]
                self.p_bests[i] = self.particles[i]
                
                # Updating best global value 
                if costs[i] < self.g_best_value:
                    self.g_best_value = costs[i]
                    self.g_best = self.particles[i]
    
    def next(self):
        """Computes an iteration of PSO"""

        if self.iter > 0:
            self.move_particles()
            self.update_bests()
            self.update_coef()

        # Updating running criterion
        self.iter += 1
        self.is_running = self.is_running and self.iter < self.max_iter and t.time()-self.start_time < self.max_time

        return self.is_running

    def move_particles(self):
        """Updates positions and velocities. Also checks if the particles are still in movement"""
        
        # Inertia
        new_velocities = self.w * self.velocities

        # Cognitive
        r_1 = np.random.random(self.N)
        r_1 = np.tile(r_1[:, None], self.size)
        new_velocities += self.cog * r_1 * (self.p_bests - self.particles)
        
        # Social
        r_2 = np.random.random(self.N)
        r_2 = np.tile(r_2[:, None], self.size)

        # Final velocities
        g_best = np.tile(self.g_best[None], (self.N, 1))
        new_velocities += self.soc * r_2 * (g_best  - self.particles)

        # Checking if there is still a movement among the particles
        self.is_running = np.sum(self.velocities - new_velocities) != 0

        # Updating positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + new_velocities
    
    def solve(self):
        """Solves the problem using PSO
        :return: a tuple containing the best found solution along with its cost"""

        while self.is_running:
            self.next()
        return self.g_best, self.g_best_value, self.particles