import argparse
import solver_naive
import solver_advanced
import solver_genetic
import GridSearch
import solver_LNS
import time
from rcpsp import RCPSP
import os


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument("--agent", type=str, default="naive")
    parser.add_argument("--infile", type=str, default="instances/instance_A_30.txt")
    parser.add_argument("--outfile", type=str, default="solution")
    parser.add_argument("--visualisation_file", type=str, default="visualization")
    parser.add_argument("--display", type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    rcpsp = RCPSP(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: RCPSP")
    print("[INFO] input file: %s" % args.infile)
    print("[INFO] output file: %s" % args.outfile)
    print("[INFO] visualization file: %s" % args.visualisation_file)
    print("[INFO] number of nodes: %s" % (rcpsp.graph.number_of_nodes()))
    print("[INFO] number of edges: %s" % (rcpsp.graph.number_of_edges()))
    print("***********************************************************")

    start_time = time.time()

    # Méthode à implémenter
    if args.agent == "naive":
        # trivial extensive search
        solution = solver_naive.solve(rcpsp)
    elif args.agent == "genetic":
        # Genetic Agent
        solution = solver_genetic.solve(rcpsp)
    elif args.agent == "lns":
        # LNS Agent
        solution = solver_LNS.solve(rcpsp)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(rcpsp)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60, 2)

    base_path = f"output/{args.agent}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # You can disable the display if you do not want to generate the visualization
    plot_name = (
        args.visualisation_file
        if args.visualisation_file != "visualization"
        else f"{base_path}/visualization_{args.infile.replace('instances/', '').replace('.txt', '')}"
    )
    rcpsp.display_solution(solution, plot_name, args.display)

    rcpsp.save_solution(
        solution,
        f"{base_path}/{args.outfile}_{args.infile.replace('instances/', '').replace('.txt', '')}",
    )
    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Makespan : %s" % rcpsp.get_solution_cost(solution))
    print("[INFO] Sanity check passed : %s" % rcpsp.verify_solution(solution))
    print("***********************************************************")
