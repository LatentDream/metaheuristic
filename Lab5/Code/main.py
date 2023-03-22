import argparse
from atelier import Atelier

import local_search as ls
import beam_search as beam
import local_beam_search as lbs
import grasp
import ils
import aco
import solver_advanced as sa

import random as r
import numpy as np

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--scenario', type=str, default="")
    parser.add_argument('--infile', type=str, default="./instances/atelier_16_20.txt")
    parser.add_argument('--outfile', type=str, default="solution.txt")
    parser.add_argument('--visufile', type=str, default="visu.png")
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--mode', type=str, default="random")

    # Local Search parameters
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--p', type=float, default=0.1)

    # (Local) Beam Search parameters
    parser.add_argument('--pop_size', type=int, default=16)

    # GRASP parameters
    parser.add_argument('--rand_fact', type=float, default=0.2)

    # ILS parameters
    parser.add_argument('--accept_temp', type=int, default=10000)
    parser.add_argument('--red_factor', type=float, default=0.995)
    parser.add_argument('--pert', type=float, default=0.3)

    # ACO parameters
    parser.add_argument('--n_ants', type=int, default=16)
    parser.add_argument('--n_travel', type=int, default=64)
    parser.add_argument('--ph_max', type=int, default=3)
    parser.add_argument('--ph_min', type=int, default=1)
    parser.add_argument('--ph_evap', type=float, default=0.2)
    parser.add_argument('--ph_rein', type=float, default=0.125)
    parser.add_argument('--hew', type=float, default=1)
    parser.add_argument('--phw', type=float, default=1)

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
    
    # Beam search
    elif args.scenario == "S2":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Beam Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] population size: %s" % args.pop_size)
        print("***********************************************************")

        solution = beam.beam_search(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    # Local beam search
    elif args.scenario == "S3":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Local Beam Search")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] population size: %s" % args.pop_size)
        print("***********************************************************")

        solution = lbs.local_beam_search(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    # Greedy Randomized Adaptative Search Procedure
    elif args.scenario == "S4":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with GRASP")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] randomization factor: %s" % args.rand_fact)
        print("***********************************************************")

        solution = grasp.grasp(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")
    
    # Iterated Local Search
    elif args.scenario == "S5":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with ILS")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] generation mode: %s" % args.mode)
        print("[INFO] maximum computation time: %s" % args.max_time)
        print("[INFO] maximum non-improving iterations: %s" % args.max_iter)
        print("[INFO] proportion of possible permutations considered: %s" % args.p)
        print("[INFO] proportion of perturbation: %s" % args.pert)
        print("[INFO] initial acceptation temperature: %s" % args.accept_temp)
        print("[INFO] reduction factor of the temperature: %s" % args.red_factor)
        print("***********************************************************")

        solution = ils.ils(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    # Ant-Colony Optimization
    elif args.scenario == "S6":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with ACO")
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

        solution = aco.aco(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    elif args.scenario == "S7":
        atelier = Atelier(args.infile)
        print("***********************************************************")
        print("[INFO] Start the solving of Machines Location Problem with Advanced Solver")
        print("[INFO] input file: %s" % args.infile)
        print("[INFO] output file: %s" % args.outfile)
        print("[INFO] visualization file: %s" % args.visufile)
        print("[INFO] number of machines: %s" % atelier.n_machines)
        print("[INFO] number of jobs: %s" % atelier.n_jobs)
        print("[INFO] niveau de connaissance de votre implémentation par le chargé de TD : %s" % 0)
        print("[INFO] Conséquence sur le plot : créez un joli plot vous-même :)")
        print("***********************************************************")

        solution = sa.solve(atelier, args)
        atelier.save_solution(solution, args)

        print("[INFO] final solution: ", solution) 
        print("[INFO] final cost: ", atelier.get_total_cost(solution))
        print("***********************************************************")

    else:
        raise Exception("Scénario non implémenté")