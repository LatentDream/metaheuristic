import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from math import inf


class TSPTW:
    def __init__(self, filename: str):
        """Initialize the TSPTW (traveling salesman with time windows) structure from input file. The resulting graph will
        have nodes with the following attributes:
        - index (int): the index of the node. INDEXES start at 1
        The graph will have edges with the following attributes:
        - weight (float): weight of the edge
        Args:
            filename (str): path to the instance file
        """
        distances = {}
        self.distance_min = inf
        self.distance_max = -1
        self.time_windows = []
        with open(filename, "r") as f:
            lines = f.readlines()
            self.num_nodes = int(lines[0])
            node_attributes = {
                index + 1: {"index": index + 1} for index in range(self.num_nodes)
            }
            for i, line in enumerate(lines):
                line = line.strip().split(" ")
                if i == 0:
                    continue
                elif i <= self.num_nodes:
                    # distances[i-1] = {j: {"weight": float(distance)} for j, distance in enumerate(line)}
                    # Modification to save max and min distance
                    dist = {}
                    for j, distance in enumerate(line):
                        distance = float(distance)
                        dist[j] = {"weight": distance}
                        self.distance_max = max(self.distance_max, distance)
                        self.distance_min = min(self.distance_min, distance)
                    distances[i - 1] = dist
                else:
                    self.time_windows.append([int(k) for k in line])

        self.graph = nx.from_dict_of_dicts(distances, create_using=nx.DiGraph())
        nx.set_node_attributes(self.graph, node_attributes)

    def display_solution(self, solution: List[int], plot_name: str = ""):
        """Display a solution as an interactive graph and save it under the name given to the
        argument "plot_name". The plot will have the following characteristics:
            - Edges included in the solution will be displayed as GREEN
            - The depot will be displayed as GREEN
            - Nodes will be labeled with their id
            - Edges will have labels corresponding to their weight
        Args:
            solution List[int]
            plot_name (str): name given to the output plot
        """
        solution_edges = []
        for i in range(len(solution) - 1):
            edge = [solution[i], solution[i + 1]]
            solution_edges.append(edge)

        pos = nx.spring_layout(self.graph, k=50, seed=0)
        node_label_dict = {i: f"{self.time_windows[i]}" for i in range(self.num_nodes)}
        nx.draw_networkx(self.graph, pos=pos, edgelist=[])
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] - 0.1)
        # , labels=node_label_dict
        nx.draw_networkx_labels(self.graph, pos_attrs, labels=node_label_dict)
        nx.draw_networkx_edges(self.graph, pos, connectionstyle="arc3,rad=0.15")
        edge_labels = {}

        edge_labels = dict(
            [
                (
                    (
                        u,
                        v,
                    ),
                    f'{int(d["weight"])}\n\n{int(self.graph.edges[(v,u)]["weight"])}',
                )
                for u, v, d in self.graph.edges(data=True)
                if pos[u][0] > pos[v][0]
            ]
        )
        nx.draw_networkx_edge_labels(
            self.graph, pos=pos, edge_labels=edge_labels, label_pos=0.3
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=solution_edges,
            width=8,
            alpha=0.5,
            edge_color="tab:green",
            connectionstyle="arc3,rad=0.15",
        )
        nx.draw_networkx_nodes(
            self.graph, pos=pos, nodelist=[0], node_color="tab:green"
        )
        plt.savefig(f"{plot_name}.png", format="PNG")
        plt.show()

    def verify_solution(self, solution: List[int]) -> bool:
        """Verifies if a solution is feasible.

        Args:
            solution (List[int]): solution

        Returns:
            solution_valid (bool): whether or not the given solution is valid:
                - all elements in the solution are different except for the depot, which appears in the first and
                  last position of the solution
                - all time windows are respected
        """
        # Solution starts and ends at depot
        if solution[0] != 0 or solution[-1] != 0:
            return False
        # All expected nodes are in the solution / allDifferent constraint
        expected_solution_nodes = {i for i in range(1, self.num_nodes)}
        for node in solution:
            if node != 0:
                expected_solution_nodes.remove(node)
        if expected_solution_nodes:
            return False
        # Time window constraints are respected
        total_time = 0
        for i in range(len(solution) - 1):
            current_node = solution[i]
            next_node = solution[i + 1]
            lower_bound, upper_bound = self.time_windows[next_node]
            total_time += self.graph[current_node][next_node]["weight"]
            total_time = max(total_time, lower_bound)
            if total_time > upper_bound:
                # print("CONFLICt current_node", i)
                # print("next", i + 1)
                return False

        return True

    def get_solution_cost(self, solution: List[int]) -> int:
        """Computes and returns the cost of a solution

        Args:
            solution (List[int]): solution in the format [0, p1, p2, ..., pn, 0] where p1, ..., pn is a permutation
            of the nodes. p1, ..., pn are all integers representing the id of the node. The solution starts and ends with 0
            as the tour starts from the depot

        Returns:
            total_weight (int): cost of the solution
        """
        total_time = 0
        for i in range(len(solution) - 1):
            current_node = solution[i]
            next_node = solution[i + 1]
            lower_bound, upper_bound = self.time_windows[next_node]
            total_time += self.graph[current_node][next_node]["weight"]
            total_time = max(total_time, lower_bound)

        return total_time

    def save_solution(self, solution: List[int], output_file: str = "") -> None:
        """Saves solution to file

        Args:
            solution (List[int]): Solution
            output_file (str, optional): name of the output file where the solution will be saved
        """
        with open(output_file, "w") as file:
            file.write(str(self.num_nodes) + "\n")
            for row in solution:
                file.write(str(row) + "\n")
