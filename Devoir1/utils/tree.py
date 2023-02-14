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
        def find(node, node_id):
            if node.id == node_id:
                return node
            for child in node.children:
                found_node = find(child, node_id)
                if found_node != None:
                    return found_node

        # Find the root: 
        root = self
        while root.parent != None:
            root = root.parent

        return find(root, node_id)

    def detach_from_parent(self):
        if self.parent == None:
            return
        old_parent = self.parent
        old_parent.children.remove(self)
        self.parent = None
        return old_parent

    def add_child(self, child):
        self.children.add(child)
        old_parent = child.detach_from_parent()
        child.parent = self
        return old_parent

    def root(self):
        """ Convert this node as the root """
        if self.parent == None: 
            return self

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

        return self
 
    def get_connection_list(self) -> Tuple[List[Tuple[int]], List[int]]:
        node = self
        while node.parent != None:
            node = node.parent

        return node.get_children_connection_list()


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

    @property
    def depth_below(self):
        """ Get depth max under this node """
        def find_depth(node, depth):
            depth += 1
            max_depth = depth
            for child in node.children:
                d = find_depth(child, depth)
                if d > max_depth:
                    max_depth = d
            return max_depth
        
        return find_depth(self, 0)


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
    print(f"Number of node: {len(pcstp.network.nodes)}")
    root_dict = random.choice(pcstp.network.nodes) #! Bug here, choice try to access index 0 ?
    root_id = root_dict['index']
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
