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

    def get_node(self, node_id):
        self.root()
        def find(node, node_id):
            if node.id == node_id:
                return node
            for child in node.children:
                found_node = find(child, node_id)
                if found_node != None:
                    return found_node

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
        
        if self.parent == None: return
        parent = self.parent
        self.children.add(parent)
        self.parent = None
        convert(parent, self)

        return self

 
    def get_connection_list(self) -> Tuple[List[Tuple[int]], List[int]]:
        self.root()

        return self.get_children_connection_list()


    def get_children_connection_list(self) -> Tuple[List[Tuple[int]], List[int]]:
        connections = set()
        nodes_id = list()
        def add_connection(node):
            for child in node.children:
                connection = (node.id, child.id) 
                # Block infinite recursive call in case of cycle
                if connection not in connections:
                    nodes_id.append(child.id)
                    connections.add(connection)
                    add_connection(child)
        nodes_id.append(self.id)
        add_connection(self)
        return list(connections), nodes_id


    def __str__(self) -> str:
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



def build_valid_solution(pcstp: PCSTP, debug:bool = False) -> Node:
    """ From a PCSTP, build a valid tree and return the connection list and the tree """
    # Connect everything
    root_id = random.choice(pcstp.network.nodes)['index']
    nodes_in_tree = {root_id}
    # Create the tree
    root_node = Node(root_id)

    def build_tree(id, parent_node):
        for adj_node in pcstp.network.adj[id]:
            if adj_node not in nodes_in_tree:
                if debug: print(f"Adding: {id} -> {adj_node}")
                child = Node(adj_node, parent_node)
                nodes_in_tree.add(adj_node)
                build_tree(adj_node, child)

    build_tree(root_id, root_node)

    return root_node
