import itertools
from typing import List
from solver_genetic import solve
from rcpsp import RCPSP


def grid_search(rcpsp: RCPSP, hyperparams):
    # Create a list of all possible combinations of hyperparameters
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    param_combinations = list(itertools.product(*param_values))

    # Evaluate each combination of hyperparameters
    results = []
    for params in param_combinations:
        print(f"Evaluating parameters: {dict(zip(param_names, params))}")
        fitness_function = lambda solution: solve(
            rcpsp, **dict(zip(param_names, params))
        )
        score = genetic_algorithm(rcpsp, fitness_function=fitness_function)
        results.append((dict(zip(param_names, params)), score))

    # Sort the results by score (ascending)
    results.sort(key=lambda x: x[1])

    return results


rcpsp = RCPSP(...)
hyperparams = {
    "pop_size": [30, 50, 100],
    "mutation_rate": [0.05, 0.1, 0.2],
    "max_iter_local_search": [100, 200, 500],
    # Add more hyperparameters and their possible values here
}
results = grid_search(rcpsp, hyperparams)
print(results)