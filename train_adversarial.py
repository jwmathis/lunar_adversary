"""
The Strategy: Defensive Hardening

In this phase, the Pilot's environment becomes a "Gauntlet."

    The Constant: Load the best saboteur and apply its specific force profile to every single trial in the NEAT population.

    The Goal: The Pilot must evolve a new "Internal Model" of physics that expects sudden lateral whiplash.

    The Reward: We keep the standard reward system, but the "Fitness" will naturally be lower at first because the Saboteur is successfully crashing the early generations.
"""

import gymnasium as gym
import neat
import os
import time
import pickle
from visualize import *
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import itertools

# Create folders if they don't exist
for folder in ['checkpoints_adversarial', 'adversarial_evolution_snapshots', 'adversarial_evolution_plots']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        
"""
Description: The NEAT trainer. Takes a group of genomes (babies) and puts them in the pilot's seat

Parameters
----------
genomes : list
    A list of genomes (babies)
config : neat.config.Config
    The configuration file
"""
SABOTEUR_PATH = 'saboteur_brain/budget_consc_saboteur.pkl'
PILOT_PATH = 'pilot_brain/best_pilot_brain.pkl'
CONFIG_PATH = 'config-feedforward'
# Global counter to track generations across calls
current_generation = 0

def eval_genomes(genomes, config):
    global current_generation
    env = gym.make("LunarLander-v3")
    try:
        with open(SABOTEUR_PATH, 'rb') as f:
            saboteur_data = pickle.load(f)
            # Access the forces array from your Saboteur object
            saboteur_forces = saboteur_data.forces 
    except FileNotFoundError:
        print("CRITICAL ERROR: best_saboteur.pkl not found. Run Phase 2 first!")
        sys.exit(1)
        
    # CURRICULUM: Scale force from 20% to 100% over 200 generations
    scaling_factor = min(1.0, 0.2 + (current_generation / 200.0))
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run_scores = []
        
        # Using 10 seeds for high-reliability testing
        seeds = [42] + [random.randint(0, 100000) for _ in range(9)] 
        
        for i, sim_seed in enumerate(seeds):
            observation, info = env.reset(seed=sim_seed)
            total_run_reward = 0
            is_adversarial = (i < 8) # 80% adversarial, 20% normal
            
            # Match the Saboteur's 600 step training window
            for step in range(600):
                # --- APPLY ADVERSARIAL FORCE ---
                # Map current step to one of the 20 force segments
                segment_index = min(step // 30, len(saboteur_forces) - 1)
                wind_force = saboteur_forces[segment_index] * scaling_factor
                
                # Apply the lateral wind force to the lander's center
                env.unwrapped.lander.ApplyForceToCenter((wind_force, 0.0), True)
                
                # --- PILOT DECISION MAKING ---
                output = net.activate(observation)
                action = output.index(max(output))
                observation, reward, terminated, truncated, info = env.step(action)

                # Variable Extraction
                x_pos, y_pos = observation[0], observation[1]
                v_horz, v_vert = observation[2], observation[3]
                angle = observation[4]
                left_leg, right_leg = observation[6], observation[7]
                dist_from_center = abs(x_pos)

                # 1. THE LAVA WALLS (Keeps the pilot centered)
                boundary_penalty = 0
                if dist_from_center > 0.8:
                    boundary_penalty = ((dist_from_center - 0.8) * 70.0) ** 2

                # 2. PRECISION CENTER MAGNET
                center_reward = 1.5 - (dist_from_center ** 2) * 8.0
                
                # 3. THE "SILENCE" LOGIC (Ground-Zone Finesse)
                if y_pos < 0.25:
                    # Harsh penalties for horizontal drift and tilt near ground
                    stability_penalty = (abs(angle) * 10.0) + (abs(v_horz) * 8.0)
                    
                    if v_vert < -0.10: 
                        stability_penalty += abs(v_vert) * 25.0 
                    
                    # --- THE TWITCH KILLER ---
                    if left_leg and right_leg:
                        if action != 0: 
                            total_run_reward -= 15.0 # Punishment for jitter
                        else:
                            total_run_reward += 25.0 # MASSIVE bonus for shutting down
                    elif left_leg or right_leg:
                        if action != 0:
                            total_run_reward -= 8.0
                else:
                    # High altitude stabilization
                    stability_penalty = (abs(angle) * 0.5) + (abs(v_horz) * 0.2)

                # 4. FLIGHT DYNAMICS
                descent_pressure = 0
                if y_pos > 0.1:
                    if v_vert < -0.35:   # Falling too fast
                        descent_pressure = -2.0
                    elif v_vert < -0.05: # Optimal glide
                        descent_pressure = 2.0
                    else:               # Wasting fuel hovering
                        descent_pressure = -1.0

                # 5. FINAL ACCUMULATION
                total_run_reward += (reward + center_reward + descent_pressure - stability_penalty - boundary_penalty)
                
                # Flat Time Tax to prevent "Hover-stalling"
                total_run_reward -= 0.10

                # 6. TERMINAL BONUSES
                if terminated or truncated:
                    if reward <= -100: # Crash
                        total_run_reward -= 400 
                    
                    if reward >= 100: # Successful Landing
                        if dist_from_center < 0.05:
                            total_run_reward += 5000.0 # Bullseye sniper
                        elif dist_from_center < 0.15:
                            total_run_reward += 2000.0
                        else:
                            total_run_reward += 1000.0 
                    break
            
            run_scores.append(total_run_reward)
        
        # Fitness is the average across all trials
        genome.fitness = sum(run_scores) / len(run_scores)
    current_generation += 1    
    env.close()
 
"""
Description: The Manager. Controls the entire experiment over days or weeks of simulated time.

Parameters
----------
config_file : str
    The configuration file
"""
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    # 1. Start with the Veteran population
    population = neat.Population(config)
    
    # Check if we have a veteran pilot to seed the population
    if os.path.exists(PILOT_PATH):
        print("\n--- PHASE 3: Hardening the Veteran Pilot ---")
        with open(PILOT_PATH, 'rb') as f:
            veteran_genome = pickle.load(f)
        
        # 1. Update the population genomes
        # Manually overwrite the DNA of every genome in the new population
        for g in population.population.values():
            # Deep copy the veteran's nodes and connections
            g.nodes = copy.deepcopy(veteran_genome.nodes)
            g.connections = copy.deepcopy(veteran_genome.connections)
            g.fitness = None 
            
        # 1. Sync Node IDs (Integers)
        max_node = max(veteran_genome.nodes.keys())

        # 2. Sync Connection IDs (Integers only)
        # We search through every connection and look for an integer ID
        innovation_numbers = []
        for conn in veteran_genome.connections.values():
            # Check if 'innovation' or 'key' is the integer ID
            if hasattr(conn, 'innovation') and isinstance(conn.innovation, int):
                innovation_numbers.append(conn.innovation)
            elif hasattr(conn, 'key') and isinstance(conn.key, int):
                innovation_numbers.append(conn.key)
        
        # If we can't find an integer ID, we fall back to a safe high number
        if innovation_numbers:
            max_conn = max(innovation_numbers)
        else:
            # Emergency fallback: use node count + 1000 to avoid collisions
            max_conn = max_node + 1000 

        # 3. Apply the Counters
        config.genome_config.node_indexer = itertools.count(max_node + 1)
        config.genome_config.connection_indexer = itertools.count(max_conn + 1)
        
        print(f"SUCCESS: Counters Synced at Nodes:{max_node+1}, Conns:{max_conn+1}")
        
        # 2. Update the population species
        population.species.speciate(config, population.population, population.generation) 
    else:
        print("\n--- WARNING: No veteran found. Starting from scratch... ---")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter() 
    population.add_reporter(stats)
    
    # 2. SEPARATE DIRECTORY: Don't overwrite Phase 1 checkpoints!
    checkpoint_dir = 'checkpoints_adversarial' 
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_prefix = os.path.join(checkpoint_dir, 'adversarial-neat-checkpoint-')
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix))

    # 3. The Run
    winner = population.run(eval_genomes, 500) 
        
    print("Training complete! Saving the Robust Champion...")
    # Change these names so you don't lose your original work
    with open('robust_final_pilot.pkl', 'wb') as f:
        pickle.dump(winner, f)
        
    best_ever = stats.best_genome() 
    print(f"Final Robust Fitness: {best_ever.fitness}")

    with open('robust_pilot_brain.pkl', 'wb') as f:
        pickle.dump(best_ever, f)
 

"""
Description: Test the best pilot saved to disk

Parameters
----------
config_path : str
    The configuration file path
genome_path : str
    The genome file path
"""
def test_best_pilot(genome_path):
    # Load Saboteur for the visualizer
    with open(SABOTEUR_PATH, 'rb') as f:
        saboteur = pickle.load(f)
        
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    with open(genome_path, 'rb') as f:
        winner_genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    env = gym.make("LunarLander-v3", render_mode="human")

    # Add a loop so it keeps running
    while True: 
        observation, info = env.reset()
        screen = pygame.display.set_mode((1000, 1000))
        terminated = False
        truncated = False
        fuel_spent = 0.0
        print("Champion is starting a new flight...")
        step = 0
        while not (terminated or truncated):
            # Apply the force so the visual matches the reality
            segment_index = min(step // 30, len(saboteur.forces) - 1)
            wind_force = saboteur.forces[segment_index]
            env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)
            
            # The 'activate' function gives us signals for the 4 engines
            outputs = net.activate(observation)
            action = outputs.index(max(outputs))
            if action == 2: # Main engine
                fuel_spent += 0.3
            elif action in [1, 3]: # Side engines
                fuel_spent += 0.03
            observation, reward, terminated, truncated, info = env.step(action)
            
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 1000))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 1000), 3)
            draw_hud(env, reward, observation, step, fuel_spent, wind_force=wind_force) # Draw the HUD
            draw_realtime_brain(winner_genome, config, observation) # Draw the brain activity
            
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
                
            pygame.draw.line(screen, color, start_pos, end_pos, 5)
            pygame.draw.circle(screen, color, end_pos, 7) # Arrow head
            
            step += 1
        
            pygame.display.flip() # refresh the screen
            
        print("Flight finished. Resetting in 2 seconds...")
        import time
        time.sleep(2) # Pause to see the final position

def playback_evolution(checkpoint_folder, config_path, interval=50):
    """
    Renders a flight for every 'interval' generations.
    """
    # 1. Setup the environment for rendering
    # Use 'human' to watch it live, or 'rgb_array' if you want to record
    env = gym.make("LunarLander-v3", render_mode="human")
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 2. Get and sort checkpoints
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('adversarial-neat-checkpoint-')]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    for filename in files:
        gen_num = int(filename.split('-')[-1])
        
        # Only play back specific milestones (e.g., every 50 generations)
        if gen_num % interval != 0 and gen_num != 373: # Always include the final one
            continue

        print(f"\n--- Showing Pilot from Generation {gen_num} ---")
        checkpoint_path = os.path.join(checkpoint_folder, filename)
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        
        # Find the best genome in this checkpoint
        best_genome = None
        for g in p.population.values():
            if best_genome is None or (g.fitness is not None and g.fitness > best_genome.fitness):
                best_genome = g
        
        if best_genome is None: continue

        # 3. Fly the mission
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        observation, info = env.reset(seed=42) # Use same seed to see improvement on same map
        screen = pygame.display.set_mode((1000, 1000))
        done = False
        total_reward = 0
        fuel_spent = 0.0
        step = 0
        while not done:
            env.render()
            output = net.activate(observation)
            action = output.index(max(output))
            # Track fuel
            if action == 2: fuel_spent += 0.3
            elif action in [1, 3]: fuel_spent += 0.03
            step += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
             # --- Visuals ---
            # Standard HUD/Brain drawing
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 1000))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 1000), 3)

            draw_hud(env, total_reward, observation, step, fuel_spent=fuel_spent, gen_label=gen_num)
            draw_realtime_brain(best_genome, config, observation)
            
            pygame.display.flip()
            
        print(f"Generation {gen_num} Result: {total_reward:.2f}")
        time.sleep(1) # Pause between generations

    env.close()
    
def validate_pilot(config_path, genome_path, num_episodes=50):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    with open(genome_path, 'rb') as f:
        winner_genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    # Use non-render mode for speed, or "human" if you want to watch all 50
    env = gym.make("LunarLander-v3") 

    successes = 0
    total_reward = 0
    crashes = 0

    print(f"\n--- Validating Pilot over {num_episodes} Random Maps ---")
    
    for i in range(num_episodes):
        observation, info = env.reset() # Random seed every time
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            outputs = net.activate(observation)
            action = outputs.index(max(outputs))
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        # In LunarLander, a reward of 200+ is a successful landing
        if episode_reward >= 100:
            successes += 1
        elif episode_reward < -100:
            crashes += 1
            
        total_reward += episode_reward
        print(f"Trial {i+1}: Reward = {episode_reward:.2f}")

    print("\n--- FINAL VALIDATION RESULTS ---")
    print(f"Success Rate: {(successes/num_episodes)*100:.1f}%")
    print(f"Crash Rate: {(crashes/num_episodes)*100:.1f}%")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    env.close()

def validate_pilot_precision(brain_path, config_path, num_trials=50):
    # Load the brain and config
    with open(brain_path, "rb") as f:
        genome = pickle.load(f)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("LunarLander-v3")
    
    rewards = []
    displacements = []
    success_count = 0

    print(f"--- Validating Precision over {num_trials} Random Maps ---")

    for i in range(num_trials):
        observation, info = env.reset() # Random seed
        done = False
        episode_reward = 0
        
        while not done:
            output = net.activate(observation)
            action = np.argmax(output)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done:
                # Track how far from x=0 the lander is at the end
                final_x = observation[0]
                displacements.append(abs(final_x))
                
                # Check for Gymnasium's official success (200 pts)
                if episode_reward >= 200:
                    success_count += 1
        
        rewards.append(episode_reward)
        print(f"Trial {i+1}: Reward = {episode_reward:>7.2f} | Final X = {final_x:>6.3f}")

    # --- FINAL SUMMARY ---
    avg_reward = sum(rewards) / num_trials
    avg_precision = sum(displacements) / num_trials
    success_rate = (success_count / num_trials) * 100
    
    print("\n" + "="*40)
    print("      FINAL PRECISION RESULTS")
    print("="*40)
    print(f"Success Rate:     {success_rate:.1f}%")
    print(f"Average Reward:   {avg_reward:.2f}")
    print(f"Average Offset:   {avg_precision:.4f} units")
    
    # Analyze the offset
    if avg_precision < 0.05:
        print("Rating: SNIPER (Inside the bullseye)")
    elif avg_precision < 0.2:
        print("Rating: PROFESSIONAL (Inside the flags)")
    else:
        print("Rating: ROOKIE (Safe but scattered)")
    print("="*40)

    env.close()
    plot_precision_histogram(displacements)
            
def make_evolution_plots():
    os.makedirs("evolution_plots", exist_ok=True)
    visualize_all_checkpoints("checkpoints_adversarial", "config-feedforward")
 
def plot_fitness_from_checkpoints(checkpoint_folder, filename="evolution_graph.png"):
    # 1. Gather all checkpoint files and sort them numerically
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('adversarial-neat-checkpoint-')]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    generations = []
    best_fitness = []
    avg_fitness = []

    print(f"Extracting data from {len(files)} checkpoints...")

    for filename in files:
        gen_num = int(filename.split('-')[-1])
        checkpoint_path = os.path.join(checkpoint_folder, filename)
        
        # Load the population snapshot
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        
        # Extract fitness values from the current population
        all_fitnesses = [g.fitness for g in p.population.values() if g.fitness is not None]
        
        if all_fitnesses:
            generations.append(gen_num)
            best_fitness.append(max(all_fitnesses))
            avg_fitness.append(sum(all_fitnesses) / len(all_fitnesses))

    # 2. Your Plotting Logic (Modified slightly to use the extracted lists)
    plt.style.use('dark_background') 
    plt.figure(figsize=(12, 6))
    
    # Plotting the data we gathered
    plt.plot(generations, best_fitness, label='Best Pilot', color='#50E3C2', linewidth=2)
    plt.plot(generations, avg_fitness, label='Average Population', color='#4A90E2', linestyle='--')
    
    # Formatting
    plt.title('Pilot Intelligence Evolution (Reconstructed from Checkpoints)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Score)')
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.savefig('evolution_graph.png')
    print(f"Graph saved as evolution_graph.png")
    plt.show()
 
def plot_smoothed_fitness(checkpoint_folder, window_size=5):
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('adversarial-neat-checkpoint-')]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    generations, best_fitness, avg_fitness = [], [], []

    for filename in files:
        p = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_folder, filename))
        all_fits = [g.fitness for g in p.population.values() if g.fitness is not None]
        if all_fits:
            generations.append(int(filename.split('-')[-1]))
            best_fitness.append(max(all_fits))
            avg_fitness.append(sum(all_fits) / len(all_fits))

    # Convert to Series for easy math
    best_series = pd.Series(best_fitness)
    avg_series = pd.Series(avg_fitness)
    
    # Calculate Moving Averages
    smoothed_best = best_series.rolling(window=window_size, min_periods=1).mean()
    smoothed_avg = avg_series.rolling(window=window_size, min_periods=1).mean()

    plt.style.use('dark_background') 
    plt.figure(figsize=(12, 6))
    
    # Plot Raw Data (faintly in the background)
    plt.plot(generations, best_fitness, color='#50E3C2', alpha=0.2, label='_nolegend_')
    plt.plot(generations, avg_fitness, color='#4A90E2', alpha=0.1, label='_nolegend_')
    
    # Plot Smoothed Data (thick and bold)
    plt.plot(generations, smoothed_best, label=f'Best Pilot ({window_size}-Gen Avg)', color='#50E3C2', linewidth=3)
    plt.plot(generations, smoothed_avg, label=f'Pop. Mean ({window_size}-Gen Avg)', color='#4A90E2', linewidth=2, linestyle='--')
    
    plt.title('Smoothed Pilot Intelligence Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smoothed_evolution.png')
    plt.show()
         
def plot_precision_histogram(displacements):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    # Create the histogram
    # We use 20 bins to get a detailed view of the 0.0 to 0.5 range
    n, bins, patches = plt.hist(displacements, bins=20, color='#50E3C2', alpha=0.7, edgecolor='white')
    
    # Highlight the "Goal Zone" (the flags are at 0.2)
    plt.axvspan(0, 0.2, color='green', alpha=0.1, label='Inside Flags (Success Zone)')
    plt.axvline(0.2, color='red', linestyle='--', alpha=0.5, label='Flag Boundary')
    
    plt.title('Landing Precision Distribution (Distance from Center)')
    plt.xlabel('Offset from Center (0.0 = Perfect Bullseye)')
    plt.ylabel('Number of Landings')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('landing_precision.png')
    print("Precision histogram saved as landing_precision.png")
    plt.show()
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    
    if len(sys.argv) < 2:
        print("Usage: python train_pilot.py [train/test_history/test_best] [optional: genome_file]")
        sys.exit(1)
    else: 
        command = sys.argv[1].lower()
        if command == "train":
            # Check if a checkpoint file was provided as the second argument
            if len(sys.argv) > 2:
                checkpoint_file = sys.argv[2]
                print(f"Resuming from checkpoint: {checkpoint_file}")

                # Restore the population state
                p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
                p.config.fitness_threshold = 12000
                p.config.conn_add_prob = 0.1
                p.config.node_add_prob = 0.05
                p.config.max_stagnation = 20
                p.config.weight_mutate_power = 0.2
                # Re-add reporters because they aren't saved in the checkpoint
                p.add_reporter(neat.StdOutReporter(True))
                stats = neat.StatisticsReporter()
                p.add_reporter(stats)
                checkpoint_dir = 'checkpoints'
                checkpoint_prefix = os.path.join(checkpoint_dir, 'neat-checkpoint-')
                p.add_reporter(neat.Checkpointer(1, filename_prefix=checkpoint_prefix))
                
                # Start running again
                winner = p.run(eval_genomes, 50) # Run for 500 more generations
                # Save the final
                print("Training complete! Saving the final champion...")
                with open('final_pilot_brain.pkl', 'wb') as f:
                    pickle.dump(winner, f)
                    
                # GET THE ABSOLUTE BEST EVER SEEN
                # This looks through the entire history, not just the last generation
                best_ever = stats.best_genome() 

                print(f"Final Best Fitness: {best_ever.fitness}")

                with open('best_pilot_brain.pkl', 'wb') as f:
                    pickle.dump(best_ever, f)
        
            else:
                # Start from scratch
                run_neat(config_path)

        elif command == "test_best":
            if len(sys.argv) < 2:
                print("Please provide the genome file to test.")
                sys.exit(1)
            genome_file = sys.argv[2]
            test_best_pilot(genome_file)
        elif command == "validate":
            if len(sys.argv) < 3:
                print("Usage: python train_pilot.py validate [genome_file]")
                sys.exit(1)
            genome_file = sys.argv[2]
            validate_pilot(config_path, genome_file)
        elif command == "validate_precision":
            if len(sys.argv) < 3:
                print("Usage: python train_pilot.py validate_precision [genome_file]")
                sys.exit(1)
            genome_file = sys.argv[2]
            validate_pilot_precision(genome_file, config_path)
        elif command == "plots":
            make_evolution_plots()
            plot_fitness_from_checkpoints('checkpoints_adversarial')
            plot_smoothed_fitness('checkpoints_adversarial', window_size=5)
        elif command == "playback":
            playback_evolution('checkpoints_adversarial', config_path=CONFIG_PATH, interval=10)
        else:
            print("Unknown command. Use 'train', 'test_best', or 'test_history'.")
            sys.exit(1)
            