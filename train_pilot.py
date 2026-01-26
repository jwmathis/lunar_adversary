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
# Create folders if they don't exist
for folder in ['checkpoints', 'evolution_snapshots', 'evolution_plots']:
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

def eval_genomes(genomes, config):
    env = gym.make("LunarLander-v3")
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run_scores = []
        
        # Using 10 seeds for high-reliability testing
        seeds = [42] + [random.randint(0, 100000) for _ in range(9)] 
        
        for sim_seed in seeds:
            observation, info = env.reset(seed=sim_seed)
            total_run_reward = 0
            
            for _ in range(1000):
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
                        total_run_reward -= 800 
                    
                    if reward >= 100: # Successful Landing
                        if dist_from_center < 0.05:
                            total_run_reward += 1000.0 # Bullseye sniper
                        elif dist_from_center < 0.15:
                            total_run_reward += 500.0 
                    break
            
            run_scores.append(total_run_reward)
        
        # Fitness is the average across all trials
        genome.fitness = sum(run_scores) / len(run_scores)
        
    env.close()
 
"""
Description: The Manager. Controls the entire experiment over days or weeks of simulated time.

Parameters
----------
config_file : str
    The configuration file
"""
def run_neat(config_file):
    # Configure NEAT and learns the rules (8 inputs, 4 outputs, population size of 50, etc)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
    population = neat.Population(config) # Creates a geneeration of pilots (genomes) completely at random, "dumb" brains
    population.add_reporter(neat.StdOutReporter(True)) # Outputs progress to the terminal
    stats = neat.StatisticsReporter() 
    population.add_reporter(stats) # Collects and reports statistics
    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'neat-checkpoint-')
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix)) # Saves progress every 10 generations

    winner = population.run(eval_genomes, 500) # Run for up to 1000 generations
        
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
 

"""
Description: Test the best pilot saved to disk

Parameters
----------
config_path : str
    The configuration file path
genome_path : str
    The genome file path
"""
def test_best_pilot(config_path, genome_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

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

        while not (terminated or truncated):
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
            draw_hud(env, reward, observation, 0, fuel_spent) # Draw the HUD
            draw_realtime_brain(winner_genome, config, observation) # Draw the brain activity
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
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('neat-checkpoint-')]
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
    visualize_all_checkpoints("checkpoints", "config-feedforward")
 
def plot_fitness_from_checkpoints(checkpoint_folder, filename="evolution_graph.png"):
    # 1. Gather all checkpoint files and sort them numerically
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('neat-checkpoint-')]
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
    files = [f for f in os.listdir(checkpoint_folder) if f.startswith('neat-checkpoint-')]
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
            if len(sys.argv) < 3:
                print("Please provide the genome file to test.")
                sys.exit(1)
            genome_file = sys.argv[2]
            test_best_pilot(config_path, genome_file)
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
            plot_fitness_from_checkpoints('checkpoints')
            plot_smoothed_fitness('checkpoints', window_size=5)
        elif command == "playback":
            playback_evolution('checkpoints', config_path, interval=10)
        else:
            print("Unknown command. Use 'train', 'test_best', or 'test_history'.")
            sys.exit(1)
            