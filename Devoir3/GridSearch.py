import itertools
from typing import List
import solver_genetic
from rcpsp import RCPSP


hyperparams = {
    "pop_size": [30, 50, 100],
    "mutation_rate": [0.05, 0.1, 0.2],
    "max_iter_local_search": [100, 200, 500],
    "tournament_size": [10, 20],
    "tournament_accepted": [2, 5],
    "num_generations": [10, 30, 50],
    "no_progress_generations": [5],
    "elite_size": [2],
}


def grid_search(rcpsp: RCPSP, hyperparams):
    # Create a list of all possible combinations of hyperparameters
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    param_combinations = list(itertools.product(*param_values))

    time_limit = 60

    # Evaluate each combination of hyperparameters
    results = []
    for pop_size in hyperparams["pop_size"]:
        for mutation_rate in hyperparams["mutation_rate"]:
            for max_iter_local_search in hyperparams["max_iter_local_search"]:
                for tournament_size in hyperparams["tournament_size"]:
                    for tournament_accepted in hyperparams["tournament_accepted"]:
                        for num_generations in hyperparams["num_generations"]:
                            for elite_size in hyperparams["elite_size"]:
                                for no_progress_generations in hyperparams[
                                    "no_progress_generations"
                                ]:
                                    print(
                                        "pop_size",
                                        pop_size,
                                        "\nmutation_rate",
                                        mutation_rate,
                                        "\nmax_iter_local_search",
                                        max_iter_local_search,
                                        "\ntournament_size",
                                        tournament_size,
                                        "\ntournament_accepted",
                                        tournament_accepted,
                                        "\nnum_generations",
                                        num_generations,
                                        "\nno_progress_generations",
                                        no_progress_generations,
                                        "\nelite_size",
                                        elite_size,
                                    )
                                    try:
                                        solution = solver_genetic.genetic_algorithm(
                                            rcpsp,
                                            num_generations=num_generations,
                                            no_progress_generations=no_progress_generations,
                                            max_iter_local_search=max_iter_local_search,
                                            elite_size=elite_size,
                                            mutation_rate=mutation_rate,
                                            tournament_size=tournament_size,
                                            tournament_accepted=tournament_accepted,
                                            pop_size=pop_size,
                                            time_limit=time_limit,
                                        )
                                        # Evaluate the solution and store the result
                                        score = rcpsp.get_solution_cost(solution)
                                        results.append(
                                            (
                                                {
                                                    "pop_size": pop_size,
                                                    "mutation_rate": mutation_rate,
                                                    "max_iter_local_search": max_iter_local_search,
                                                    "tournament_size": tournament_size,
                                                    "tournament_accepted": tournament_accepted,
                                                    "num_generations": num_generations,
                                                    "no_progress_generations": no_progress_generations,
                                                    "elite_size": elite_size,
                                                },
                                                score,
                                            )
                                        )
                                        print(results)
                                    except ValueError:
                                        continue

    # Sort the results by score
    results.sort(key=lambda x: x[1])

    return results


def solve(rcpsp: RCPSP) -> List[int]:
    results = grid_search(rcpsp, hyperparams)
    print(results)
    return 0
