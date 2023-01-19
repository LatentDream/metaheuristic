from knapsack import Knapsack
from utils import timer
from queue import PriorityQueue
from math import inf

# @timer
def greedy_knapsack(knapsack_instance: Knapsack):
    """ réalise la résolution gloutonne du problème du sac à dos
    :param knapsack_instance: l'instance du problème du sac à dos
    :return: la valeur totale du sac à dos obtenu
    """
    
    n_elem, max_weight = knapsack_instance.n, knapsack_instance.c
    item_sorted = PriorityQueue()
    curr_weight = 0
    curr_value = 0

    print(f"\n Problem with: {n_elem} with max {max_weight}")

    #? O(nlog(n))
    for v, w in knapsack_instance.items:
        item_sorted.put((-v/w if w != 0 else -inf, (v, w)))
    
    #? O(n)
    while True:
        _, item = item_sorted.get()
        v, w = item
        if curr_weight + w <= max_weight:
            print(f"Adding element: {v}v of weight {w}" )
            curr_weight += w
            curr_value  += v

        if curr_weight == max_weight or item_sorted.empty():
            break

    return curr_value
