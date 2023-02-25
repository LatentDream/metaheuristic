"""
    Guillaume Blanché : 2200151
    Guillaume Thibault : 1948612
"""
from typing import List, Tuple
from network import PCSTP
import random
from utils.tree import build_valid_solution, Node
from math import inf
import signal
import time

TIME_LIMIT = 30 * 60
class TimeoutException(Exception): pass
def handler(sig, frame):
    raise TimeoutException

def solve(pcstp):

    return solve_with_restart(pcstp, 1)

def solve_with_restart(pcstp, n_restart):
    best_sol = None
    best_score = inf
  
    starting_time = time.time()
    try: 
        for i in range(n_restart):
            print(f"Restart {i+1}/{n_restart} ---------- ")
            try:
                sol, stopped = solver(pcstp, starting_time)
                sol_score = pcstp.get_solution_cost(sol)
                print(f"Socre: {sol_score}")
                if sol_score < best_score:
                    best_score = sol_score
                    print(sol)
                    best_sol = sol
                if stopped:
                    raise stopped
            except KeyError as e:
                print(f"Random.choice Indexation error in build initial solution for {i+1}, restarting ...")
                continue
    except TimeoutException as e:
        print("Reason: Out of time.")
    except KeyboardInterrupt as e:
        print("Interruption requested by user.")

    return best_sol


def solver(pcstp: PCSTP, starting_time: float) -> List[Tuple[int]]:
    """Advanced solver for the prize-collecting Steiner tree problem.

    Args:
        pcstp (PCSTP): object containing the graph for the instance to solve

    Returns:
        solution (List[Tuple[int]]): contains all pairs included in the solution. For example:
                [(1, 2), (3, 4), (5, 6)]
            would be a solution where edges (1, 2), (3, 4) and (5, 6) are included and all other edges of the graph 
            are excluded
    """
    ##! Local search heuristique
    node  = build_valid_solution(pcstp, 0.5)
    connections, _ = node.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    best_solution_found, best_score = node.copy(), current_score

    nb_try_in_batch = min(len(pcstp.network.nodes)**2, 50000)
    solution_changed_in_batch = False
    nb_batch_done = 1
    nb_try = 0

    stopped = None

    print(f"\n\n{nb_try}/{nb_try_in_batch} - {solution_changed_in_batch} - {current_score}")
    print("STARTING LOCAL SEARCH \n")
    try:
        while True:

            node, change_made = find_better_local_solution(node, pcstp)
            solution_changed_in_batch |= change_made
            nb_try += 1
            current_score = pcstp.get_solution_cost(node.get_connection_list()[0])
            
            if current_score < best_score:
                best_solution_found, best_score = node.copy(), current_score
            if nb_try % 1000 == 0: 
                print(f"{nb_try}/{nb_try_in_batch} - {solution_changed_in_batch} - {current_score}")
            
            if nb_try >= nb_try_in_batch:
                if not solution_changed_in_batch:
                    print(f"Number of batch done: {nb_batch_done}")
                    break  
                else:
                    #? Reduce batch size
                    nb_try_in_batch = len(pcstp.network.nodes)**2 // nb_batch_done
                    solution_changed_in_batch = False
                    nb_batch_done += 1
                    nb_try = 0

            if time.time() - starting_time > TIME_LIMIT:
                raise TimeoutException()
    
    except (KeyboardInterrupt, TimeoutException) as e:
        print(f"\nStopping execution and returning best solution seen ...") 
        stopped = e

    connections, nodes_id = best_solution_found.get_connection_list()
    return connections, stopped
    


def find_better_local_solution(node: Node, pcstp: PCSTP) -> Tuple[Node, bool]:
    """ Try to find a better solution in a random neighborhood 
    return: 
        @Node: node from a tree with the choosing solution
        @bool: True if it's a better solution
    """
    connections, nodes_id = node.get_connection_list()
    node_id = random.choice(nodes_id)
    root = node.get_node(node_id).root()
    heuristic = random.choice([small_neighborhood_heuristic, large_neighborhood_heuristic, terminal_node_heuristic])

    return heuristic(root, pcstp)


def terminal_node_heuristic(node: Node, pcstp: PCSTP):
    """ Try to find a better solution in a neighborhood in at the bottom of the tree
    return: 
        @Node: node from a tree to execute the heuristic in
        @bool: True if it's a better solution
    """
    
    node = node.get_random_terminal_node()

    if node.parent == None:
        return node, False

    #? Remove the terminal node if it's worth it
    connections, _ = node.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    old_parent = node.detach_from_parent()
    new_connections, _ = old_parent.get_connection_list()
    new_score = pcstp.get_solution_cost(new_connections)
    if  new_score < current_score or random.random() > 0.99: #* stochasticity
        return old_parent, True
    else:
        old_parent.add_child(node)
        return old_parent, False


def large_neighborhood_heuristic(root: Node, pcstp: PCSTP)  -> Tuple[Node, bool]:
    node, change_made, neighborhood = neighborhood_heuristic(root, pcstp)
    for node in neighborhood:
        _, nodes_id = node.get_connection_list()
        if node.id in nodes_id:
            node, _, _ = neighborhood_heuristic(node.root(), pcstp)
    return node, change_made


def small_neighborhood_heuristic(root: Node, pcstp: PCSTP)  -> Tuple[Node, bool]:
    node, change_made, _ = neighborhood_heuristic(root, pcstp)
    return node, change_made


def neighborhood_heuristic(root: Node, pcstp: PCSTP)  -> Tuple[Node, bool, List[Node]]:
    """ Try to find a better solution in a neighborhood in the middle of the tree
    return: 
        @Node: node from a tree to execute the heuristic in
        @bool: True if it's a better solution
    """
    connections, nodes_id = root.get_connection_list()
    current_score = pcstp.get_solution_cost(connections)
    neighborhood = list()
    change_made = False

    for adj_node_id in pcstp.network.adj[root.id]:

        #? Case: Try to add a node if it's not already in the tree
        if adj_node_id not in nodes_id:
            new_child = Node(adj_node_id, parent=root)
            new_connections, new_nodes_id = root.get_connection_list()
            new_score = pcstp.get_solution_cost(new_connections)
            if  new_score < current_score or random.random() > 0.975: #* stochasticity
                change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                neighborhood.append(new_child)
            else:
                root.children.remove(new_child)
                new_child.parent = None
        else:
            #? Find the node in the children
            adj_node = root.get_node(adj_node_id)
            
            #? If node is attached to the current root -> Remove it
            if adj_node in root.children:
                root.children.remove(adj_node)
                new_connections, new_nodes_id = root.get_connection_list()
                new_score = pcstp.get_solution_cost(new_connections)
                if new_score < current_score and adj_node.depth_below < 5:
                    change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                else:
                    root.children.add(adj_node)

            #? If the node is attached to another node
            else:
                old_parent = adj_node.detach_from_parent()
                root.add_child(adj_node)
                new_connections, new_nodes_id = root.get_connection_list()
                new_score = pcstp.get_solution_cost(new_connections)
                if new_score < current_score:
                    change_made, connections, nodes_id, current_score = True, new_connections, new_nodes_id, new_score
                else:
                    adj_node.detach_from_parent()
                    old_parent.add_child(adj_node)
                neighborhood.append(adj_node)


    return root, change_made, neighborhood