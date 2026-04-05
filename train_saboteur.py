import gymnasium as gym
import numpy as np
import pickle
import neat
from saboteur import Saboteur, crossover, mutate
import pygame
import os
import sys
import random

# Import shared visualization tools
from visualize import draw_hud

# --- CONFIGURATION FOR TRAINING-----
PILOT_BRAIN_PATH = 'pilot_brain/best_pilot_brain.pkl'
PILOT_CONFIG_PATH = 'config-feedforward'
POP_SIZE = 50
GENERATIONS = 200
SIM_STEPS = 600  # Max steps per landing

# --- SETTINGS FOR VISUALIZATION -----
SABOTEUR_PATH = 'best_saboteur.pkl'
SEED = 1010 

# 1. Load the "Champion" Pilot trained by NEAT
try:
    with open(PILOT_BRAIN_PATH, 'rb') as f:
        pilot_genome = pickle.load(f)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Pilot brain not found at {PILOT_BRAIN_PATH}")
    sys.exit(1)
    
# Load Pilot configuration
pilot_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           PILOT_CONFIG_PATH)
pilot_net = neat.nn.FeedForwardNetwork.create(pilot_genome, pilot_config)

def evaluate_adversarial_generation(population):
    env = gym.make('LunarLander-v2')
    seeds = [42] + [random.randint(0, 100000) for _ in range(9)] 
    
    for saboteur in population:
        # Test each saboteur over multiple seeds to ensure it's "Skill" not "Luck"
        trial_scores = []
        
        for seed in seeds:
            observation, info = env.reset(seed=seed)
            score = 0
            
            for step in range(SIM_STEPS):
                # 1. Pilot decides action
                output = pilot_net.activate(observation)
                action = output.index(max(output))
                
                # 2. Saboteur applies force
                segment_index = min(step // (SIM_STEPS // len(saboteur.forces)), len(saboteur.forces) - 1)
                wind_force = saboteur.forces[segment_index]
                
                # Apply the wind force to the environment
                env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
                
                observation, reward, terminated, truncated, info = env.step(action)
                score += reward
                
                if terminated or truncated:
                    break
            trial_scores.append(score)
            
        # Saboteur's Fitness is how much the Pilot Failed and if it uses minimal force
        avg_force_used = np.mean(np.abs(saboteur.forces))
        peak_force_used = np.max(np.abs(saboteur.forces))
        saboteur.fitness = (- np.mean(trial_scores)) - (avg_force_used * 5.0) - (peak_force_used * 4.0)
        
    env.close()

def train_saboteur():
    population = [Saboteur(num_segments=10) for _ in range(POP_SIZE)]
    
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
    best_saboteur = population[0]
    avg_magnitude = np.mean(np.abs(best_saboteur.forces))
    peak_force = np.max(np.abs(best_saboteur.forces))
    print(f"--- Saboteur Efficiency Metrics ---")
    print(f"Average Force Magnitude: {avg_magnitude:.2f}")
    print(f"Peak Force Magnitude: {peak_force:.2f}")
    print(f"Efficiency Score: {best_saboteur.fitness / max(1.0, avg_magnitude):.2f}")

def visualize_sabotage():
    # 1. Load the Best Saboteur
    if not os.path.exists(SABOTEUR_PATH):
        print(f"Error: {SABOTEUR_PATH} not found.")
        return

    with open(SABOTEUR_PATH, 'rb') as f:
        saboteur = pickle.load(f)
        
    print("\n--- ATTACK PROFILE ---")
    print(f"{'Segment':<10} | {'Step Range':<15} | {'Force (N)':<10} | {'Description'}")
    print("-" * 60)
    for i, force in enumerate(saboteur.forces):
        start_step = i * (SIM_STEPS // len(saboteur.forces))
        end_step = (i + 1) * (SIM_STEPS // len(saboteur.forces))
        
        if abs(force) < 2: desc = "Quiet"
        elif abs(force) < 8: desc = "Nudge"
        elif abs(force) < 15: desc = "Strong Wind"
        else: desc = "CRITICAL STRIKE"
        
        direction = "Right →" if force > 0 else "Left  ←"
        print(f"{i:<10} | {start_step:>4}-{end_step:<10} | {force:>8.2f}N {direction} | {desc}")
    print("-" * 60 + "\n")
       
    # 2. Setup Environment
    env = gym.make('LunarLander-v2', render_mode='human')
    observation, info = env.reset(seed=SEED)
    
    total_reward = 0
    total_force_spent = 0.0 
    steps = 0
    max_steps = SIM_STEPS
    
    print(f"---- Monitoring Saboteur vs Pilot on Seed {SEED} ----")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- PILOT DECISION ---
        output = pilot_net.activate(observation)
        action = output.index(max(output))
        
        # --- SABOTEUR DECISION ---
        segment_index = min(steps // (max_steps // len(saboteur.forces)), len(saboteur.forces) - 1)
        wind_force = saboteur.forces[segment_index]
        
        total_force_spent += abs(wind_force)
        env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # --- DRAWING THE METRICS OVERLAY ---
        canvas = pygame.display.get_surface()
        if canvas is not None:
            # Use shared HUD
            draw_hud(env, total_reward, observation, steps, 0, wind_force)
            
            font = pygame.font.SysFont("Arial", 22)
            efficiency_ratio = (0 - total_reward) / max(1.0, total_force_spent)
            avg_force = total_force_spent / max(1, steps)

            eff_text = font.render(f"Efficiency Ratio: {efficiency_ratio:.2f}", True, (255, 215, 0))
            avg_text = font.render(f"Avg Force Used: {avg_force:.2f}N", True, (255, 215, 0))
            
            canvas.blit(eff_text, (20, 220))
            canvas.blit(avg_text, (20, 250))
            
            # --- DRAW THE FORCE VECTOR (Arrow) ---
            start_pos = (400, 100) 
            end_pos = (400 + int(wind_force * 5), 100)
            
            if abs(wind_force) > 15: color = (255, 50, 50)     
            elif abs(wind_force) > 5: color = (200, 0, 255)    
            else: color = (200, 200, 200)   
                
            pygame.draw.line(canvas, color, start_pos, end_pos, 5)
            pygame.draw.circle(canvas, color, end_pos, 7) 
            pygame.display.flip()
            
        if terminated or truncated:
            final_eff = (0 - total_reward) / max(1.0, total_force_spent)
            print(f"\n--- MISSION DEBRIEF ---")
            print(f"Final Score: {total_reward:.2f}")
            print(f"Total Steps: {steps}")
            print(f"Average Force: {total_force_spent/steps:.2f}N")
            print(f"Efficiency Score: {final_eff:.4f}")
            running = False
            
    env.close()
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_saboteur.py [train/test]")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    if command == "train":
        train_saboteur()
    elif command == "test":
        visualize_sabotage()