import numpy as np

class Saboteur:
    def __init__(self, num_segments=10):
        # A list of 10 random forces between -15 and 15
        self.forces = np.random.uniform(-15, 15, num_segments)
        self.fitness = 0

def crossover(parent1, parent2):
    # Mix the forces of two successful saboteurs
    child = Saboteur()
    mask = np.random.rand(10) > 0.5
    child.forces = np.where(mask, parent1.forces, parent2.forces)
    return child

def mutate(saboteur, rate=0.1):
    # Occasionally tweak a force value
    for i in range(len(saboteur.forces)):
        if np.random.rand() < rate:
            saboteur.forces[i] += np.random.normal(0, 2.0)