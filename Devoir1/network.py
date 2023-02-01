"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""

import bokeh
from bokeh.transform import transform
from bokeh.models import BoxZoomTool, Circle, CustomJSTransform, LabelSet, HoverTool, MultiLine, Plot, Range1d, ResetTool
from bokeh.plotting import from_networkx
import networkx as nx
import pandas as pd
from typing import List, Set, Tuple
import matplotlib.pyplot as plt


class PCSTP():
    def __init__(self, filename: str):
        """Initialize the PCSTP (prize collecting steiner tree) structure from input file. The resulting graph will
        have nodes with the following attributes:
        - weight (float): the weight of the node. If the node is non-terminal, its weight will be 0
        - index (int): the index of the node. INDEXES start at 1
        The graph will have edges with the following attributes:
        - weight (float): weight of the edge
        
        Args:
            filename (str): path to the instance file
        """
        self.num_terminal_nodes = 0
        self.network = nx.Graph()
        with open(filename, "r") as f:
            lines = f.readlines()
            num_nodes = int(lines[0].split(" ")[0])
            node_attributes = {index + 1: {"index": index+1} for index in range(num_nodes)}
            for line in lines:
                line = line.split(" ")
                if line[0] == "E":
                    origin, destination, edge_weight = int(
                        line[1]), int(line[2]), float(line[3])
                    self.network.add_edge(
                        origin, destination, weight=edge_weight)
                elif line[0] == "T":
                    self.num_terminal_nodes += 1
                    node_id, node_weight = int(line[1]), float(line[2])
                    node_attributes[node_id]["terminal"] = 1
                    node_attributes[node_id]["weight"] = node_weight

            nx.set_node_attributes(self.network, 0, "terminal")
            nx.set_node_attributes(self.network, 0, "weight")
            nx.set_node_attributes(self.network, node_attributes)

    def display_solution(self, solution: Set[Tuple[int]], plot_name: str = ""):
        """Display a solution as an interactive graph and save it under the name given to the 
        argument "plot_name". The plot will have the following characteristics:
            - Edges included in the solution will be displayed as GREEN
            - Edges excluded in the solution will be displayed as RED
            - Terminal nodes included in the solution will be displayed as GREEN
            - Terminal nodes excluded from the solution will be displayed as RED
            - Non-terminal nodes will be displayed as BLUE
            - Nodes will have labels i (for index) and w (for weight)
            - Edges will have labels corresponding to their weight
        Args:
            solution (Set[Tuple[int]]) : contains all pairs included in the solution. For example:
                  [(1, 2), (3, 4), (5, 6)]
                would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph 
                are excluded
            plot_name (str): name given to the output plot
        """
        pos = nx.spring_layout(self.network, k=5, seed=0)
        labeldict = {}
        for node_id, node_attributes in self.network.nodes(data=True):
            if node_attributes["terminal"]:
                node_label = f"i: {node_id} \n w:{int(node_attributes['weight'])}"
            else:
                node_label = ""
            labeldict[node_id] = node_label

        nx.draw(self.network, font_size=8, labels=labeldict,
                with_labels=True, pos=pos)
        edge_labels = {(edge[0], edge[1]): int(edge[2]["weight"]) for edge in self.network.edges(data=True)}
        nx.draw_networkx_edge_labels(self.network, pos=pos, edge_labels=edge_labels)
        
        included_edges = []
        excluded_edges = []
        for edge in self.network.edges():
            if edge in solution:
                included_edges.append(edge)
            else:
                excluded_edges.append(edge)
        nx.draw_networkx_edges(
            self.network,
            pos,
            edgelist=included_edges,
            width=8,
            alpha=0.5,
            edge_color="tab:green",
        )
        nx.draw_networkx_edges(
            self.network,
            pos,
            edgelist=excluded_edges,
            width=8,
            alpha=0.5,
            edge_color="tab:red",
        )

        included_terminals, excluded_terminals, non_terminal = self.get_node_colors(solution)

        nx.draw_networkx_nodes(self.network, node_size=900, pos=pos, nodelist=excluded_terminals, node_color="tab:red")
        nx.draw_networkx_nodes(self.network, node_size=900, pos=pos, nodelist=included_terminals, node_color="tab:green")
        nx.draw_networkx_nodes(self.network, node_size=900, pos=pos, nodelist=non_terminal, node_color="tab:blue")

        plt.savefig(f"{plot_name}.png", format="PNG")
        plt.show()

    def get_node_colors(self, solution: Set[Tuple[int]]) -> Tuple[List[int]]:
        """Returns a dict mapping node_id to node color for visualisation
        Args:
            solution (Set[Tuple[int]]): solution to prize-collecting Steiner tree problem
        Returns:
            included_terminals (List[int]): terminal nodes included in the solution
            excluded_terminals (List[int]): terminal nodes excluded from the solution
            non_terminal (List[int]): non-terminal nodes
        """
        included_terminals = []
        excluded_terminals = []
        non_terminal = []
        
        all_solution_nodes = set()
        for edge in solution:
            all_solution_nodes = all_solution_nodes.union(set(edge))

        for node_id, node_attributes in self.network.nodes(data=True):
            if node_attributes["terminal"]:
                if node_id in all_solution_nodes:
                    included_terminals.append(node_id)  # terminal nodes included in the solution are displayed as green
                else:
                    excluded_terminals.append(node_id)  # terminal nodes excluded from the solution are displayed as red
            else:
                # non terminal nodes are displayed as blue
                non_terminal.append(node_id)

        return included_terminals, excluded_terminals, non_terminal

    def verify_solution(self, solution: Set[Tuple[int]]) -> bool:
        """Verifies if a solution is feasible.

        Args:
            solution (Set[Tuple[int]]): solution

        Returns:
            solution_valid (bool): whether or not the given solution is valid:
                - no cycle
                - all edges belong to the original graph
                - solution has one connected component
        """
        solution_graph = nx.Graph(solution)
        # no cycle
        try:
            has_no_cycle = not nx.find_cycle(solution_graph)
        except:
            has_no_cycle = True
        # only one connected component
        has_one_connected_component = nx.number_connected_components(solution_graph) == 1
        # all solution edges are in the original graph
        all_solution_edges_in_original_graph = True
        sorted_original_edges = sorted(self.network.edges())
        for solution_edge in sorted(solution_graph.edges()):
            all_solution_edges_in_original_graph = all_solution_edges_in_original_graph and (solution_edge in sorted_original_edges)

        return all_solution_edges_in_original_graph and has_no_cycle and has_one_connected_component
    
    def get_solution_cost(self, solution: Set[Tuple[int]]) -> int:
        """Computes and returns the cost of a solution

        Args:
            solution (Set[Tuple[int]]): solution

        Returns:
            total_weight (int): cost of the solution
        """
        solution_graph = nx.Graph(solution)
        total_weight = 0
        for node in solution_graph.nodes():
            node_weight = self.network.nodes[node]["weight"]
            total_weight += node_weight

        for edge in solution_graph.edges():
            edge_weight = self.network.edges[edge]["weight"]
            total_weight += edge_weight
        
        return total_weight
    
    def save_solution(self, solution: Set[Tuple[int]], output_file:str="") -> None:
        """Saves solution to file

        Args:
            solution (Set[Tuple[int]]): Solution
            output_file (str, optional): name of the output file where the solution will be saved
        """
        solution = list(solution)
        solution = sorted(solution, key=lambda x: x[0])
        pd.DataFrame(solution).to_csv(output_file + ".csv", header=False)