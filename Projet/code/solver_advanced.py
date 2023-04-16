"""
Guillaume Blanché
Guillaume Thibault
"""

import solver_genetic
import solver_lns
import solver_heuristic_layer

def solve_advanced(e):
    """
    Your solver for the problem
    :param e: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    solution = solver_heuristic_layer.solve_heuristic(e)[0]
    return solver_lns.lns(e,solution,search_time=20*60)
    return solver_genetic.solve_advanced(e)
