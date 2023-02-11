from typing import List, Tuple
from network import PCSTP
import random

class Node:

    def __init__(self, id: str, parent=None) -> None:
        """ Constructor """
        self.id = id
        self.children = set()
        self.parent = parent
        if parent != None: self.parent.children.add(self)


    def root(self):
        """ Convert this node as the root """
        def convert(node, child_that_became_parent):
            parent = node.parent
            # Remove new parent from node child
            node.children.remove(child_that_became_parent)
            # Add new parent
            node.parent = child_that_became_parent
            # Go update node higher in the tree
            if parent:
                node.children.add(parent)
                convert(parent, node)
        
        parent = self.parent
        self.children.add(parent)
        self.parent = None
        convert(parent, self)

    def connection_list():
        raise NotImplementedError()

    def __str__(self):
        """ Pretty print the tree """
        def recursive_print(output, node, prefix):
            output += f"{prefix}|_{node.id} \n"
            prefix += "  "
            for child in node.children:
                output = recursive_print(output, child, prefix)
            return output
        
        output = ""
        if self.parent != None:
            output += ".\n.\n.\n"
        return recursive_print(output, self, "")


def build_valid_solution(pcstp: PCSTP, debug:bool = False) -> Tuple[List[Tuple[int]], Node]:
    """ From a PCSTP, build a valid tree and return the connection list and the tree """
    # TODO: Remove the tree_connection_list var when the connection_list() is implemented
    # Connect everything
    root_id = random.choice(pcstp.network.nodes)['index']
    root_id = 2
    nodes_in_tree = {root_id}
    # Create a list of connection
    tree_connection_list = list()
    # Create a real tree
    root_node = Node(root_id)

    def build_tree(id, parent_node):
        for adj_node in pcstp.network.adj[id]:
            if adj_node not in nodes_in_tree:
                if debug: print(f"Adding: {id} -> {adj_node}")
                tree_connection_list.append((id, adj_node))
                child = Node(adj_node, parent_node)
                nodes_in_tree.add(adj_node)
                build_tree(adj_node, child)

    build_tree(root_id, root_node)

    return root_node, tree_connection_list
    