"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 
"""

from typing import List, Tuple
from network import PCSTP
import networkx as nx


def solve(pcstp: PCSTP) -> List[Tuple[int]]:
    """Naive solution to the problem: connect 2 terminal nodes via shortest path.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph
            are excluded
    """
    terminal_nodes = []
    for node_id, node_attributes in pcstp.network.nodes(data=True):
        if node_attributes["terminal"]:
            terminal_nodes.append(node_id)
        if len(terminal_nodes) >= 2:
            break
    solution_nodes = nx.shortest_path(
        pcstp.network, terminal_nodes[0], terminal_nodes[1]
    )
    solution = []
    for i in range(1, len(solution_nodes)):
        origin = solution_nodes[i - 1]
        destination = solution_nodes[i]
        edge = (origin, destination)
        solution.append(edge)

    return solution
