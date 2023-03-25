import random


import random

import random


import random


def m_point_crossover(parent1, parent2, m):
    # Ensure m is between 0 and len(parent1) - 1
    m = max(min(m, len(parent1) - 1), 0)

    # Choose m random crossover points
    crossover_points = random.sample(range(1, len(parent1)), m)

    # Sort the crossover points
    crossover_points.sort()

    # Initialize offspring chromosomes
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    # Perform crossover between the parents at the crossover points
    for i in range(0, m, 2):
        start_point = crossover_points[i]
        if i + 1 < m:
            end_point = crossover_points[i + 1]
        else:
            end_point = len(parent1)

        # Swap the corresponding segments of the parents
        offspring1[start_point:end_point], offspring2[start_point:end_point] = (
            offspring2[start_point:end_point],
            offspring1[start_point:end_point],
        )

    # Repair offspring chromosomes by replacing repeated elements with missing elements
    offspring1 = repair_chromosome(offspring1, parent1, parent2)
    offspring2 = repair_chromosome(offspring2, parent2, parent1)

    return offspring1, offspring2


def repair_chromosome(chromosome, parent1, parent2):
    # Get the set of missing elements in the offspring chromosome
    missing_elements = list(set(parent1) - set(chromosome))

    while len(missing_elements) > 0:
        # Get the set of repeated elements in the offspring chromosome
        repeated_elements = set([x for x in chromosome if chromosome.count(x) > 1])
        # Replace repeated elements with missing elements
        for i in range(len(chromosome)):
            if chromosome[i] in repeated_elements:
                chromosome[i] = missing_elements[0]
                missing_elements = missing_elements[1:]
                repeated_elements = set(
                    [x for x in chromosome if chromosome.count(x) > 1]
                )

        missing_elements = list(set(parent1) - set(chromosome))

    return chromosome


parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
parent2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
m = 4

print(parent1)
print(parent2)
offspring1, offspring2 = m_point_crossover(parent1, parent2, m)
print(offspring1)  # Output: [1, 2, 3, 7, 6, 5, 4, 3, 9, 10]
print(offspring2)  # Output: [10, 9, 8, 4, 5, 6, 7, 8, 2, 1]
