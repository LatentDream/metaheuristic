from math import *
import argparse
import random as r
from pso import PSO
import numpy as np
import utils
import copy

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=2)
    parser.add_argument('--n_particles', type=int, default=10)
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--auto_coef', type=bool, default=True)

    parser.add_argument('--x_min', type=tuple, default=(-10,-10))
    parser.add_argument('--x_max', type=tuple, default=(10,10))
    parser.add_argument('--v_min', type=tuple, default=(-20,-20))
    parser.add_argument('--v_max', type=tuple, default=(20,20))

    parser.add_argument('--ine', type=float, default=0.8)
    parser.add_argument('--cog', type=float, default=2)
    parser.add_argument('--soc', type=float, default=2)

    parser.add_argument('--step', type=float, default=5)

    return parser.parse_args()

if __name__ == '__main__':

    r.seed(10)
    np.random.seed(10)
    args = parse_arguments()

    print("***********************************************************")
    print("[INFO] Start the solving of the minimization problem using PSO")
    print("[INFO] maximum computation time: %s" % args.max_time)
    print("[INFO] maximum number of iterations: %s" % args.max_iter)
    print("[INFO] number of particles: %s" % args.n_particles)
    print("[INFO] inertia coefficient: %s" % args.ine)
    print("[INFO] cognitive coefficient: %s" % args.cog)
    print("[INFO] social coefficient: %s" % args.soc)
    print("[INFO] dynamic coefficients: %s" % args.auto_coef)
    print("[INFO] minimal coordinates: %s" % list(args.x_min))
    print("[INFO] maximal coordinates: %s" % list(args.x_max))
    print("[INFO] minimal velocities: %s" % list(args.v_min))
    print("[INFO] maximal velocities: %s" % list(args.v_max))
    print("***********************************************************")

    fitness_function = utils.fitness_function
    solution_random, cost_random = utils.get_random_sol(args, fitness_function)
    
    particles, velocities = utils.generate_initial_swarm(args)
    particles_in = copy.copy(particles)
    model = PSO(particles, velocities, fitness_function, 
                w=args.ine, cog=args.cog, soc=args.soc, max_iter=args.max_iter, max_time=args.max_time, auto_coef=args.auto_coef)
    solution_pso, cost_pso, particles_out = model.solve()

    print("[INFO] random solution: ", solution_random) 
    print("[INFO] random cost: ", cost_random)
    print("[INFO] PSO solution: ", solution_pso) 
    print("[INFO] PSO cost: ", cost_pso)
    print("***********************************************************")
    utils.plot_function(args, particles_in,particles_out)
