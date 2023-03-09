import argparse
from atelier import Atelier

import local_search as ls
import simulated_annealing as sa
import tabu_search as ts
import tabu_search_advanced as ts_advanced
import genetic_search as gs
import genetic_search_advanced as gs_advanced

import random as r
import numpy as np

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--scenario', type=str, default="")
    parser.add_argument('--infile', type=str, default="./instances/atelier_7_9.txt")
    parser.add_argument('--outfile', type=str, default="solution.txt")
    parser.add_argument('--visufile', type=str, default="visu.png")
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--mode', type=str, default="random")

    # Local Search parameters
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--max_restarts', type=int, default=20)

    # Simulated Annealing parameters
    parser.add_argument('--init_temp', type=int, default=10000)
    parser.add_argument('--red_factor', type=float, default=0.95)
    
    # Tabu parameters
    parser.add_argument('--mu', type=int, default=15)
    parser.add_argument('--sigma', type=float, default=5.0)

    # Genetic parameters
    parser.add_argument('--max_gen', type=int, default=1000)
    parser.add_argument('--pop_size', type=int, default=64)
    parser.add_argument('--mut_rate', type=float, default=0.1)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    r.seed(10)
    np.random.seed(10)

    # Simple local search
    if args.scenario == "S1":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Local Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("***********************************************************")

        solution = ls.local_search(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    # Local search with restarts
    elif args.scenario == "S2":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Local Search and Restarts")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] maximum number of restarts: %s" % args.max_iter)
        print("***********************************************************")

        solution = ls.restarts(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    # Simulated annealing
    elif args.scenario == "S3":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Simulated Annealing Local Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] initial temperature: %s" % args.init_temp)
        print("[INFO] temperature reduction factor: %s" % args.red_factor)
        print("***********************************************************")

        solution = sa.simulated_annealing(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    # Tabu search
    elif args.scenario == "S4":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Tabu Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("")
        print("[INFO] Number of tabu iteration for a permutation is determined using a standard law")
        print("[INFO] expectance of standard law : %s" % args.mu)
        print("[INFO] standard deviation of standard law: %s" % args.sigma)
        print("***********************************************************")

        solution = ts.tabu_search(atelier, args)
        atelier.save_solution(solution, args)
        atelier.display_solution(solution,args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    # Tabu search advanced
    elif args.scenario == "S5":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Advanced Tabu Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] niveau de connaissance de votre implémentation par le chargé de TD : %s" % 0)
        print("[INFO] Conséquence sur le plot : créez un joli plot vous-même :)")
        print("***********************************************************")

        solution = ts_advanced.tabu_search_advanced(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    # Genetic search
    elif args.scenario == "S6":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Genetic Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] population size: %s" % args.pop_size)
        print("[INFO] maximum number of generations: %s" % args.max_gen)
        print("[INFO] selection type: roulette")
        print("[INFO] crossing type: UX")
        print("[INFO] mutation type: 10% permutation")
        print("***********************************************************")

        solution = gs.genetic_search(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    # Genetic search advanced
    elif args.scenario == "S7":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Advanced Genetic Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] niveau de connaissance de votre implémentation par le chargé de TD : %s" % 0)
        print("[INFO] Conséquence sur le plot : créez un joli plot vous-même :)")
        print("***********************************************************")

        solution = gs_advanced.solve(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    else:
        raise Exception("Scénario non implémenté")