import gymnasium as gym
import numpy as np
import pickle
import neat

# 1. Load the "Champion" Pilot trained by NEAT
with open('best_pilot_brain', 'rb') as f:
    champion_genome = pickle.load(f)
    
def eval_saboteurs(genomes, config):
    # Load the Pilot's brain configuration
    with open('config-feedforward', 'r') as f:
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    f)

    # Create the environment
    env = gym.make("LunarLander-v3")
    
    for genome_id, genome in genomes:
        # Saboteur's "brain" is just a list of forces
        forces = genome.vector
        
        observation, info  = env_reset()
        pilot_score = 0
        
        for step in range(500):
            # Pilot decides what to do
            action = pilot_net.activate(observation).index(max(output))
            
            # Saboteur applies the force
            force_index = min(step // 50, len(forces) - 1)
            wind_force = forces[force_index]
            
            # Apply force directly to the physics engine
            env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0, 0), True)
            
            observation, reward, terminated, truncated, info = env.step(action)
            pilot_score += reward
            
            if terminated or truncated:
                break
        # Success for Saboteur = Failure for Pilot
        # Want the Saboteur's fitness to be high when the Pilot's score is low
        # Update the fitness score
        genome.fitness = -pilot_score
        
    env.close()