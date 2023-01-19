import copy
from math import inf

def dynamic_knapsack_recursive(knapsack_instance):
    """réalise la résolution par programmation dynamique récursive
    :param knapsack_instance: l'instance du problème du sac à dos
    :param item: le numéro du dernier item considéré
    :param cap: la capacité maximale considérée
    #? Garanty of the optimal solution - But slower O(n^2)
    """

    def recursive(knapsack):
        if knapsack.c == 0:
            return 0
        if knapsack.n == 0:
            return 0
        
        # Get item to test
        v, w = knapsack.items[0]

        #* Option don't choose item
        knapsack_copy = copy.deepcopy(knapsack)
        knapsack_copy.items = knapsack_copy.items[1:]
        knapsack_copy.n -= 1
        v2 = recursive(knapsack_copy)

        if knapsack.c + v >= 0:
            #* Option choose item
            knapsack_copy = copy.deepcopy(knapsack)
            knapsack_copy.items = knapsack_copy.items[1:]
            knapsack_copy.c -= w
            knapsack_copy.n -= 1
            v1 = recursive(knapsack_copy)
            return max(v1 + v, v2)
            
        return v2

    return recursive(knapsack_instance)