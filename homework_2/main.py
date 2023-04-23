import argparse
import solver_naive
import solver_advanced
import solver_genetic
import solver_beam_search
import time
from tsptw import TSPTW


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument("--agent", type=str, default="naive")
    parser.add_argument("--infile", type=str, default="instances/A_4.txt")
    parser.add_argument("--outfile", type=str, default="solution")
    parser.add_argument("--visualisation_file", type=str, default="visualization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    tsptw = TSPTW(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: TSPTW")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visualisation_file)
    print("[INFO] number of nodes: %s" % (tsptw.graph.number_of_nodes()))
    print("[INFO] number of edges: %s" % (tsptw.graph.number_of_edges()))
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # trivial extensive search
        solution = solver_naive.solve(tsptw)
    elif args.agent == "genetic":
        # Genetic algorithm
        solution = solver_genetic.solve(tsptw)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(tsptw)
    elif args.agent == "beam_search":
        # More advance beam search
        solution = solver_beam_search.solve(tsptw)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60, 2)

    # You can disable the display if you do not want to generate the visualization
    tsptw.display_solution(solution, args.visualisation_file)
    #
    tsptw.save_solution(solution, args.outfile)
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(
        "[INFO] Penality obtained (value to minimize) : %s"
        % tsptw.get_solution_cost(solution)
    )
    print("[INFO] Sanity check passed : %s" % tsptw.verify_solution(solution))
    print("***********************************************************")
