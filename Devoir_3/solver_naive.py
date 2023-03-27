import time
from typing import List
from rcpsp import RCPSP
import networkx as nx
import itertools

def solve(rcpsp: RCPSP) -> List[int]:
    """Naive solution to the problem: connect 2 terminal nodes via shortest path.

    Args:
        rcpsp (RCPSP): object containing the graph for the instance to solve

    Returns:
        solution (dict): mapping between task id and start time
    """
    solution = {}
    current_time = 0

    # Sort tasks by their order in the graph, assuming the tasks are numbered sequentially
    tasks_sorted = sorted(list(rcpsp.graph.nodes))

    for task in tasks_sorted:
        solution[task] = current_time
        current_time += rcpsp.graph.nodes[task]["duration"]

    return solution