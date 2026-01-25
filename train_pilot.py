import gymnasium as gym
import neat
import os
import time
import pickle
from visualize import *
import sys


# Create folders if they don't exist
for folder in ['checkpoints', 'evolution_snapshots', 'brains']:
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
        
        # Track fitness across 3 different maps
        run_scores = []
        seeds = [42, 1337, 2024] # Specific seeds to ensure consistency
        
        for sim_seed in seeds:
            observation, info = env.reset(seed=sim_seed)
            total_run_reward = 0
            
            for _ in range(1000):
                output = net.activate(observation)
                action = output.index(max(output))
                observation, reward, terminated, truncated, info = env.step(action)

                # --- Reward Shaping (Calculated per frame) ---
                dist_from_center = abs(observation[0])
                center_reward = (1.0 - dist_from_center) * 0.1
                angle_penalty = abs(observation[4]) * 0.2
                dist_from_ground = abs(observation[1])
                v_speed = observation[3]
                ground_proximity_reward = (1.0 - dist_from_ground) * 0.2
                survival_bonus = 0.01
                
                total_run_reward += (reward + center_reward - angle_penalty + survival_bonus + ground_proximity_reward)

                # Speed penalty near ground
                if dist_from_ground < 0.2 and abs(v_speed) > 0.1:
                    total_run_reward -= (abs(v_speed) * 0.3)
                    
                if terminated or truncated:
                    break
            
            run_scores.append(total_run_reward)
        
        # The genome's fitness is the AVERAGE of all 3 maps
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

    winner = population.run(eval_genomes, 1000) # Run for up to 1000 generations
        
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

def test_evolution_history(config_path, snapshot_folder):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 1. Get all the saved brains and sort them by generation number
    # Assumes files are named 'pilot_gen_0.pkl', 'pilot_gen_10.pkl', etc.
    files = [f for f in os.listdir(snapshot_folder) if f.endswith('.pkl')]
    # Sort them numerically so Gen 0 comes before Gen 10
    files.sort(key=lambda x: int(x.split('_gen_')[1].split('.')[0]))

    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()
    screen = pygame.display.set_mode((1000, 1000))
    # Loop through history
    for filename in files:
        gen_label = filename.split('_gen_')[1].split('.')[0]
        filepath = os.path.join(snapshot_folder, filename)

        with open(filepath, 'rb') as f:
            genome = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        observation, info = env.reset()
        total_reward = 0.0
        fuel_spent = 0.0
        step = 0
        terminated = False
        truncated = False

        print(f"Now showing Pilot from Generation {gen_label}")
        
        while not (terminated or truncated):
            outputs = net.activate(observation)
            action = outputs.index(max(outputs))
            
            # Track fuel
            if action == 2: fuel_spent += 0.3
            elif action in [1, 3]: fuel_spent += 0.03
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # --- Visuals ---
            # Standard HUD/Brain drawing
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 1000))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 1000), 3)

            draw_hud(env, total_reward, observation, step, fuel_spent=fuel_spent, gen_label=gen_label)
            draw_realtime_brain(genome, config, observation)
            
            pygame.display.flip()

        #print(f"Gen {gen_label} flight finished. Next pilot in 2 seconds...")
        time.sleep(2)
        #input("Press Enter to continue...")

    env.close()
    print("Evolution Tour Complete!")

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
            # if len(sys.argv) > 2:
            #     checkpoint_file = sys.argv[2]
            #     print(f"Resuming from checkpoint: {checkpoint_file}")
            #     # Restore the population state
            #     p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            #     # Re-add reporters because they aren't saved in the checkpoint
            #     p.add_reporter(neat.StdOutReporter(True))
            #     stats = neat.StatisticsReporter()
            #     p.add_reporter(stats)
            #     p.add_reporter(neat.Checkpointer(50, filename_prefix='neat-checkpoint-'))
                
            #     # Start running again
            #     winner = p.run(eval_genomes, 500) # Run for 500 more generations
            # else:
            #     print(f"DEBUG: Loading config from: {os.path.abspath(config_path)}")
            #     # Start from scratch
            #     run_neat(config_path)
            run_neat(config_path)
        elif command == "test_best":
            if len(sys.argv) < 3:
                print("Please provide the genome file to test.")
                sys.exit(1)
            genome_file = sys.argv[2]
            test_best_pilot(config_path, genome_file)
        elif command == "test_history":
            test_evolution_history(config_path, 'evolution_snapshots')
        elif command == "validate":
            if len(sys.argv) < 3:
                print("Usage: python train_pilot.py validate [genome_file]")
                sys.exit(1)
            genome_file = sys.argv[2]
            validate_pilot(config_path, genome_file)
        else:
            print("Unknown command. Use 'train', 'test_best', or 'test_history'.")
            sys.exit(1)
            