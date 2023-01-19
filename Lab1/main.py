import os
import time
from knapsack import Knapsack
import matplotlib.pyplot as plt
import argparse
import solver_dynamic
import solver_greedy
import solver_advanced


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--agent", type=str, default=None)

    return parser.parse_args()


def get_results(size, agent_type):
    """Renvoie les résultats pour toutes les méthodes pour toutes les instances d'une certaine taille.
    size contient un string 'small' pour les petites instances ou 'medium' pour les moyennes"""

    results = {}
    times = {}
    for directory in [size]:
        d = "./instances/" + directory
        for file in os.listdir(d):
            f = os.path.join(d, file)
            name = "%s_%s" % (file.split("_")[-2], file.split("_")[-1])
            k = Knapsack(f)
            t1 = time.time()

            if agent_type == "greedy":
                solution = solver_greedy.greedy_knapsack(k)
            elif agent_type == "dynamic":
                solution = solver_dynamic.dynamic_knapsack_recursive(k)
            elif agent_type == "advanced":
                solution = solver_advanced.advanced_knapsack(k)
            else:
                raise Exception("Unknow agent")

            t2 = time.time()

            results[name] = {"solution_profit": solution}
            times[name] = {"execution_time": t2 - t1}

    return times, results


def plot_all_results(size):

    agent_list = ["greedy", "dynamic", "advanced"]
    files = []

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    if args.size == "medium":
        ax1.set_yscale("log")
    ax2.set_yscale("log")

    for agent_type in agent_list:
        t, r = get_results(size, agent_type)

        files = [a for a in t]

        times = [a["execution_time"] for a in t.values()]
        results = [a["solution_profit"] for a in r.values()]

        ax1.plot(files, times, label=agent_type)
        ax2.plot(files, results, label=agent_type)

    ax1.set_title("Comparaison des temps")
    ax1.legend()

    ax2.set_title("Comparaison des valeurs")
    ax2.legend()
    plt.show()

    return


if __name__ == "__main__":

    args = parse_arguments()

    if args.agent == "experiments":
        plot_all_results(args.size)
    else:
        print(get_results(args.size, args.agent))
