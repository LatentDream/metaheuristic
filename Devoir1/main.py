"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""

import argparse
import solver_naive
import solver_advanced
import solver_advanced_atlas
import solver_advanced_hernes
import time
from network import PCSTP


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--infile', type=str,
                        default='instances/reseau_A_8_11_5.txt')
    parser.add_argument('--outfile', type=str, default='solution')
    parser.add_argument('--visualisation_file', type=str, default='visualization')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    pcstp = PCSTP(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: network design")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visualisation_file)
    print("[INFO] number of nodes: %s" % (pcstp.network.number_of_nodes()))
    print("[INFO] number of edges: %s" % (pcstp.network.number_of_edges()))
    print("[INFO] number of terminal nodes: %s" % (pcstp.num_terminal_nodes))
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # assign a different time slot for each course
        solution = solver_naive.solve(pcstp)
    elif args.agent == "atlas":
        solution = solver_advanced_atlas.solve(pcstp)
    elif args.agent == "hernes":
        solution = solver_advanced_hernes.solve(pcstp)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(pcstp)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60,2)

    pcstp.display_solution(solution, args.visualisation_file)
    pcstp.save_solution(solution, args.outfile)
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Penality obtained (value to minimize) : %s" % pcstp.get_solution_cost(solution))
    print("[INFO] Sanity check passed : %s" % pcstp.verify_solution(solution))
    print("***********************************************************")