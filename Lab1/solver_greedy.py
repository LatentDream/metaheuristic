def greedy_knapsack(knapsack_instance):
    """réalise la résolution gloutonne du problème du sac à dos
    :param knapsack_instance: l'instance du problème du sac à dos
    :return: la valeur totale du sac à dos obtenu
    """

    cumulated_capacity = 0
    cumulated_value = 0
    selected_items = []
    capacity = knapsack_instance.c
    items_sorted = sorted(knapsack_instance.items, key=lambda x: x[0] / x[1])

    for u, w in items_sorted:
        if w + cumulated_capacity < capacity:
            selected_items.append((u, w))
            cumulated_capacity += w
            cumulated_value += u

    return cumulated_value
