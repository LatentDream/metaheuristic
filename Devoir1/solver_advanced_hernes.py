"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""
from typing import List, Tuple
from network import PCSTP
import random
from utils.tree import build_valid_solution, Node


def solve(pcstp: PCSTP, seed=0) -> List[Tuple[int]]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph 
            are excluded
    """
    if seed: 
        random.seed(seed) #! Still schtocastic 

    ######! Local search heuristique
    ##? Starting with a arbitrary VALID solution
    node  = build_valid_solution(pcstp) 
    connections, _ = node.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    print(f"Starting from: {current_score}")

    ##? Init param for nb of local search before a break check
    nb_try_in_batch = len(pcstp.network.nodes)**2
    changed_in_batch = False
    nb_batch_done = 1
    nb_try = 0

    ##? As long as there is a solution in the neighborhoods
    while True:

        ##? Change the solution locally
        node, change_made = find_better_local_solution(node, pcstp)
        changed_in_batch |= change_made
        nb_try += 1
        if nb_try % 1000 == 0: 
            print(f"{nb_try}/{nb_try_in_batch} - {changed_in_batch} - {pcstp.get_solution_cost(node.get_connection_list()[0])}")
        
        ##? Check if better when batch ended
        if nb_try >= nb_try_in_batch:
            if not changed_in_batch:
                print(f"Number of batch done: {nb_batch_done}")
                break
            ##? Reduce batch size  
            else:
                nb_try_in_batch = len(pcstp.network.nodes)**2 // nb_batch_done
                changed_in_batch = False
                nb_batch_done += 1
                nb_try = 0
        
    ##? Retourner s_i
    connections, nodes_id = node.get_connection_list()
    return connections
    


def find_better_local_solution(node: Node, pcstp: PCSTP) -> Tuple[Node, bool]:
    """ Try to find a better solution in a random neighborhood 
    return: 
        @Node: node from a tree with the choosing solution
        @bool: True if it's a better solution
    """
    #? Choose a random node in the solution
    connections, nodes_id = node.get_connection_list()
    node_id = random.choice(nodes_id)

    #? Rebuild a representation of the tree with `node_id` as the root
    root = node.get_node(node_id).root()

    # heuristic = random.choice([inner_node_heuristic, terminal_node_heuristic])
    heuristic = inner_node_heuristic

    return heuristic(root, pcstp)


def terminal_node_heuristic(node: Node, pcstp: PCSTP):
    """ Try to find a better solution in a neighborhood in at the bottom of the tree
    return: 
        @Node: node from a tree to execute the heuristic in
        @bool: True if it's a better solution
    """
    #? Find a terminal node: 
    while len(node.children) > 0:
        # choose a random children
        children_id = [child.id for child in node.children]
        node_id = random.choice(children_id)
        for child in node.children:
            if child.id == node_id:
                node = child
                break

    #? Remove the terminal node if it's worth it
    connections, _ = node.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    old_parent = node.detach_from_parent()
    new_connections, _ = old_parent.get_connection_list()
    new_score = pcstp.get_solution_cost(new_connections)
    if  new_score < current_score or random.random() > 0.999: #* stochasticity ?
        return old_parent, True
    else:
        old_parent.add_child(node)
        return old_parent, False



def inner_node_heuristic(root: Node, pcstp: PCSTP)  -> Tuple[Node, bool]:
    """ Try to find a better solution in a neighborhood in the middle of the tree
    return: 
        @Node: node from a tree to execute the heuristic in
        @bool: True if it's a better solution
    """
    connections, nodes_id = root.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    change_made = False

    #? Add or remove connection from the root to his children
    for adj_node_id in pcstp.network.adj[root.id]:

        #? Case: Try to add a node
        if adj_node_id not in nodes_id:
            #? Add the new node
            new_child = Node(adj_node_id, root)
            #? Check if the solution is better -> Keep or delete node
            new_connections, new_nodes_id = root.get_connection_list()
            new_score = pcstp.get_solution_cost(new_connections)
            if  new_score < current_score or random.random() > 0.999: #* stochasticity ?
                change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                # print(f"Adding {adj_node_id} ...")
            else:
                root.children.remove(new_child)

        #? Case: Delete a node already in the tree -> Chop chop a branch
        else:
            #? Find the node in the children
            adj_node = root.get_node(adj_node_id)
            
            #? If node is attached to the current root -> Remove it
            if adj_node in root.children:
                root.children.remove(adj_node)
                #? Check if the solution is better -> chop chop the tree or put the node back
                new_connections, new_nodes_id = root.get_connection_list()
                new_score = pcstp.get_solution_cost(new_connections)
                #! Too easy to remove node: Add a limitation on the branch of the branch the algo can chop chop
                if new_score < current_score and adj_node.depth_below < 5:
                    change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                    # print(f"Removing {adj_node_id} ...")
                else:
                    root.children.add(adj_node)

            #? If the node is attached to another node
            else:
                old_parent = adj_node.detach_from_parent()
                root.add_child(adj_node)
                #? Check if the solution is better
                new_connections, new_nodes_id = root.get_connection_list()
                new_score = pcstp.get_solution_cost(new_connections)
                if new_score < current_score:
                    change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                    # print(f"Changing child parent for {adj_node_id} ...")
                else:
                    adj_node.detach_from_parent()
                    old_parent.add_child(adj_node)

    return root, change_made