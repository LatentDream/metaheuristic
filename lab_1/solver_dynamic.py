import copy


def dynamic_knapsack_recursive(knapsack_instance):
    """réalise la résolution par programmation dynamique récursive
    :param knapsack_instance: l'instance du problème du sac à dos
    :param item: le numéro du dernier item considéré
    :param cap: la capacité maximale considérée"""

    if knapsack_instance.n == 0:
        return 0

    if knapsack_instance.c == 0:
        return 0

    u, w = knapsack_instance.items[-1]
    capacity = knapsack_instance.c

    if capacity - w > 0:
        knapsack_instance1 = copy.deepcopy(knapsack_instance)
        knapsack_instance1.n = knapsack_instance1.n - 1
        knapsack_instance1.items = knapsack_instance1.items[:-1]
        v1 = dynamic_knapsack_recursive(knapsack_instance1)

        knapsack_instance2 = copy.deepcopy(knapsack_instance)
        knapsack_instance2.n = knapsack_instance2.n - 1
        knapsack_instance2.c = knapsack_instance2.c - w
        knapsack_instance2.items = knapsack_instance2.items[:-1]
        v2 = dynamic_knapsack_recursive(knapsack_instance2)

        return max(v1, v2 + u)

    else:
        knapsack_instance.n = knapsack_instance.n - 1
        knapsack_instance.items = knapsack_instance.items[:-1]
        return dynamic_knapsack_recursive(knapsack_instance)
