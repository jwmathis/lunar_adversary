import gymnasium as gym
import numpy as np
import pickle
import neat
from saboteur import Saboteur, crossover, mutate
import pygame
import os
import sys
import random

# --- CONFIGURATION FOR TRAINING-----
PILOT_BRAIN_PATH = 'pilot_brain/best_pilot_brain.pkl'
PILOT_CONFIG_PATH = 'config-feedforward'
POP_SIZE = 50
GENERATIONS = 200
SIM_STEPS = 600  # Max steps per landing

# --- SETTINGS FOR VISUALIZATION -----
#SABOTEUR_PATH = 'best_saboteur.pkl'
SABOTEUR_PATH = 'saboteur_brain/budget_consc_saboteur.pkl'
SEED = 1010 

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
    seeds = [42] + [random.randint(0, 100000) for _ in range(9)] 
    
    for saboteur in population:
        # Test each saboteur over 3 different seeds to ensure it's "Skill" not "Luck"
        trial_scores = []
        
        for seed in seeds:
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
            
        # Saboteur's Fitness is how much the Pilot Failed and  if it uses minimal force
        avg_force_used = np.mean(np.abs(saboteur.forces))
        peak_force_used = np.max(np.abs(saboteur.forces))
        force_penalty = avg_force_used * 5.0  # Penalize high force to encourage subtlety
        saboteur.fitness = (- np.mean(trial_scores)) - (avg_force_used * 5.0) - (peak_force_used * 4.0)
        
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
    best_saboteur = population[0]
    avg_magnitude = np.mean(np.abs(best_saboteur.forces))
    peak_force = np.max(np.abs(best_saboteur.forces))
    print(f"--- Saboteur Efficiency Metrics ---")
    print(f"Average Force Magnitude: {avg_magnitude:.2f}")
    print(f"Peak Force Magnitude: {peak_force:.2f}")
    print(f"Efficiency Score: {best_saboteur.fitness / avg_magnitude:.2f}")
    
    

def visualize_sabotage():
    # 1. Load the Pilot (NEAT)
    if not os.path.exists(PILOT_BRAIN_PATH):
        print(f"Error: {PILOT_BRAIN_PATH} not found.")
        return
        
    with open(PILOT_BRAIN_PATH, 'rb') as f:
        pilot_genome = pickle.load(f)
    
    pilot_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               PILOT_CONFIG_PATH)
    pilot_net = neat.nn.FeedForwardNetwork.create(pilot_genome, pilot_config)
    
    # 2. Load the Best Saboteur (Custom GA profile)
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
        
        # Categorize the force for easy reading
        if abs(force) < 2: desc = "Quiet"
        elif abs(force) < 8: desc = "Nudge"
        elif abs(force) < 15: desc = "Strong Wind"
        else: desc = "CRITICAL STRIKE"
        
        direction = "Right →" if force > 0 else "Left  ←"
        print(f"{i:<10} | {start_step:>4}-{end_step:<10} | {force:>8.2f}N {direction} | {desc}")
    print("-" * 60 + "\n")
       
    # 3. Setup Environment
    env = gym.make('LunarLander-v3', render_mode='human')
    observation, info = env.reset(seed=SEED)
    
    total_reward = 0
    total_force_spent = 0.0  # Tracks the "energy" the saboteur uses
    steps = 0
    max_steps = SIM_STEPS
    
    print(f"---- Monitoring Saboteur vs Pilot on Seed {SEED} ----")
    
    running = True
    while running:
        # Handle Pygame events (allows you to close the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- PILOT DECISION ---
        output = pilot_net.activate(observation)
        action = output.index(max(output))
        
        # --- SABOTEUR DECISION ---
        # Map current time step to the force segments
        segment_index = min(steps // (max_steps // len(saboteur.forces)), len(saboteur.forces) - 1)
        wind_force = saboteur.forces[segment_index]
        
        # Accumulate total force spent for the efficiency calculation
        total_force_spent += abs(wind_force)
        
        # Apply the wind force directly to the lander's center of mass
        env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
        
        # Step physics
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # --- DRAWING THE METRICS OVERLAY ---
        canvas = pygame.display.get_surface()
        if canvas is not None:
            # Initialize Font
            font = pygame.font.SysFont("Arial", 22)
            
            # Calculate live metrics
            # Efficiency = Damage dealt per unit of force spent
            efficiency_ratio = (0 - total_reward) / max(1.0, total_force_spent)
            avg_force = total_force_spent / max(1, steps)

            # Draw text labels
            score_text = font.render(f"Pilot Score: {int(total_reward)}", True, (255, 255, 255))
            force_text = font.render(f"Current Force: {wind_force:.2f}N", True, (0, 255, 255))
            
            # Efficiency metrics in Gold
            eff_text = font.render(f"Efficiency Ratio: {efficiency_ratio:.2f}", True, (255, 215, 0))
            avg_text = font.render(f"Avg Force Used: {avg_force:.2f}N", True, (255, 215, 0))
            
            canvas.blit(score_text, (20, 20))
            canvas.blit(force_text, (20, 50))
            canvas.blit(eff_text, (20, 80))
            canvas.blit(avg_text, (20, 110))
            
            # --- DRAW THE FORCE VECTOR (Arrow) ---
            start_pos = (400, 100) # Top middle of the screen
            # Scale the force for visual representation
            end_pos = (400 + int(wind_force * 5), 100)
            
            # Color coding the arrow
            if abs(wind_force) > 15:
                color = (255, 50, 50)     # Bright Red: "The Bully"
            elif abs(wind_force) > 5:
                color = (200, 0, 255)    # Purple: "The Surgeon"
            else:
                color = (200, 200, 200)   # Grey: "The Breeze"
                
            pygame.draw.line(canvas, color, start_pos, end_pos, 5)
            pygame.draw.circle(canvas, color, end_pos, 7) # Arrow head
            
            pygame.display.flip()
            
        if terminated or truncated:
            # Final terminal summary
            final_eff = (0 - total_reward) / max(1.0, total_force_spent)
            print(f"\n--- MISSION DEBRIEF ---")
            print(f"Final Score: {total_reward:.2f}")
            print(f"Total Steps: {steps}")
            print(f"Average Force: {total_force_spent/steps:.2f}N")
            print(f"Efficiency Score: {final_eff:.4f}")
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
    