import gymnasium as gym
import numpy as np
import pickle
import neat
from saboteur import Saboteur, crossover, mutate
import pygame
import os
import sys

# --- CONFIGURATION FOR TRAINING-----
PILOT_BRAIN_PATH = 'pilot_brain/best_pilot_brain.pkl'
PILOT_CONFIG_PATH = 'config-feedforward'
POP_SIZE = 50
GENERATIONS = 100
SIM_STEPS = 600  # Max steps per landing

# --- SETTINGS FOR VISUALIZATION -----
SABOTEUR_PATH = 'best_saboteur.pkl'
SEED = 121

# 1. Load the "Champion" Pilot trained by NEAT
with open(PILOT_BRAIN_PATH, 'rb') as f:
    pilot_genome = pickle.load(f)
    
# Load Pilot configuration
pilot_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           PILOT_CONFIG_PATH)
pilot_net = neat.nn.FeedForwardNetwork.create(pilot_genome, pilot_config)

def evaluate_adversarial_generation(population):
    env = gym.make('LunarLander-v3')
    
    for saboteur in population:
        # Test each saboteur over 3 different seeds to ensure it's "Skill" not "Luck"
        trial_scores = []
        
        for seed in [42, 101, 999]:
            observation, info = env.reset(seed=seed)
            score = 0
            
            for step in range(SIM_STEPS):
                # 1. Pilot decides action
                output = pilot_net.activate(observation)
                action = output.index(max(output))
                
                # 2. Saboteur applies force (based on 10-segment force list)
                # Maps 600 steps to 10 force segments (60 steps each)
                segment_index = min(step // (SIM_STEPS // len(saboteur.forces)), len(saboteur.forces) - 1)
                wind_force = saboteur.forces[segment_index]
                
                # Apply the wind force to the environment
                env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
                
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                
                if terminated or truncated:
                    break
            trial_scores.append(score)
            
        # Saboteur's Fitness is how much the Pilot Failed
        # High fitness means the pilot crashed or missed the pad
        saboteur.fitness = - np.mean(trial_scores)
        
    env.close()
    


def train_saboteur():
    population = [Saboteur() for _ in range(POP_SIZE)]
    
    print(f"Training the Saboteur vs the Pilot Brain from '{PILOT_BRAIN_PATH}'")
    print(f"Starting Adversarial Training for {GENERATIONS} Generations...")
    print(f"Population Size: {POP_SIZE}, Simulation Steps per Trial: {SIM_STEPS}")
    print("-----------------------------------------------------")
    
    for generation in range(GENERATIONS):
        # 1. Evaluate
        evaluate_adversarial_generation(population)
        
        # 2. Sort by fitness (high fitness = more damage to pilot)
        population.sort(key=lambda s: s.fitness, reverse=True)
        
        best_saboteur = population[0]
        print(f"Gen {generation+1:03d} | Best Saboteur Fitness (Negative Pilot Score): {best_saboteur.fitness:.2f}")
        
        # 3. Selection and Reprodcution
        # Top 20% are selected as parents
        elites = population[:int(POP_SIZE * 0.2)]
        next_generation = elites[:] # Elitism keep the best
        
        while len(next_generation) < POP_SIZE:
            parent1, parent2 = np.random.choice(elites, 2, replace=False)
            child = crossover(parent1, parent2)
            mutate(child, rate=0.2)
            next_generation.append(child)
            
        population = next_generation
        
    # Save the best Saboteur after training
    with open('best_saboteur.pkl', 'wb') as f:
        pickle.dump(population[0], f)
    print("Adversarial Training Complete. Best Saboteur saved to 'best_saboteur.pkl'.")


def visualize_sabotage():
    # 1. Load the Pilot (NEAT)
    with open(PILOT_BRAIN_PATH, 'rb') as f:
        pilot_genome = pickle.load(f)
    
    pilot_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               PILOT_CONFIG_PATH)
    pilot_net = neat.nn.FeedForwardNetwork.create(pilot_genome, pilot_config)
    
    # 2. Load the Best Saboteur (custom GA profile)
    with open(SABOTEUR_PATH, 'rb') as f:
        saboteur = pickle.load(f)
        
    # 3. Setip Gym Environment
    env = gym.make('LunarLander-v3', render_mode='human')
    observation, info = env.reset(seed=SEED)
    
    total_reward = 0
    steps = 0
    max_steps = SIM_STEPS
    
    print(f"---- Monitoring Saboteur on Seed {SEED} ----")
    
    running = True
    while running:
        # Pilot's turn
        output = pilot_net.activate(observation)
        action = output.index(max(output))
        
        # Saboteur's turn
        segment_index = min(steps // (max_steps // len(saboteur.forces)), len(saboteur.forces) - 1)
        wind_force = saboteur.forces[segment_index]
        
        # Apply the wind force to the environment
        env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
        
        # Step physics
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # DRAWING THE VECTOR OVERLAY
        canvas = pygame.display.get_surface()
        if canvas is not None:
            # 1. Draw text (Score and Force)
            font = pygame.font.SysFont("Arial", 24)
            score_text = font.render(f"Pilot Score: {int(total_reward)}", True, (255, 255, 255))
            force_text = font.render(f"Saboteur Force: {wind_force:.2f}", True, (0, 255, 255))
            canvas.blit(score_text, (20, 20))
            canvas.blit(force_text, (20, 50))
            
            # 2. Draw force vector on lander
            start_pos = (400, 100)
            end_pos = (400 + int(wind_force * 5), 100)
            
            color = (255, 50, 50) if abs(wind_force) > 10 else (200, 200, 200)
            pygame.draw.line(canvas, color, start_pos, end_pos, 5)
            
            pygame.draw.circle(canvas, color, end_pos, 7)
            pygame.display.flip()
            
        if terminated or truncated:
            print(f"Trial finished. Final Score: {total_reward:.2f} in {steps} steps.")
            running = False
            
    env.close()
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    
    if len(sys.argv) < 2:
        print("Usage: python train_saboteur.py [train/test]")
        sys.exit(1)
    else: 
        command = sys.argv[1].lower()
        if command == "train":
            train_saboteur()
        elif command == "test":
            visualize_sabotage()
    