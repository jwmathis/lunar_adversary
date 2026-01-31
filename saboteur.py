import numpy as np

class Saboteur:
    def __init__(self, num_segments=20):
        # A list of 20 random forces between -20 and 20
        self.forces = np.random.uniform(-20, 20, num_segments)
        self.fitness = 0

def crossover(parent1, parent2):
    # Mix the forces of two successful saboteurs
    num_segments = len(parent1.forces)
    child = Saboteur(num_segments=num_segments)
    mask = np.random.rand(num_segments) > 0.5
    child.forces = np.where(mask, parent1.forces, parent2.forces)
    return child

def mutate(saboteur, rate=0.2):
    # Occasionally tweak a force value
    for i in range(len(saboteur.forces)):
        if np.random.rand() < rate:
            # 20% chance of setting the force to zero (turning off the force)
            if np.random.rand() < 0.2:
                saboteur.forces[i] = 0.0
            else:
                # Add a small gaussian "nudge" to the force
                saboteur.forces[i] += np.random.normal(0, 2.0)
                # Clip to keep forces within reasonable physical limits
                saboteur.forces[i] = np.clip(saboteur.forces[i], -25.0, 25.0)