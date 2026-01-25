import gymnasium as gym
import neat
import os
import time
import pickle
from visualize import *


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
    # Opens the simulator and Create the environment (non-render mode is faster for training)
    env = gym.make("LunarLander-v3")
    
    # Iterate through each individual in the population
    for genome_id, genome in genomes:
        # Take the "DNA" (the genome) and Create the "brain" (the neural network)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Reset the environment
        observation, info = env.reset() # Places the lander at the top of the screen and gets the 8 pieces of data (coordinates, velocity, angle, etc)
        terminated = False
        total_reward = 0

        # Run the simulation 
        for _ in range(500): # Limit number of steps to prevent infinite hovering
            # Predict action based on the 8 lander observations
            output = net.activate(observation) # Feed the observations to the network and gives us signals for the 4 engines
            action = output.index(max(output)) # choose the highest signal as the action 
            observation, reward, terminated, truncated, info = env.step(action) # Apply the action

            total_reward += reward

            #time.sleep(0.01)
            
            if terminated or truncated:
                break

        # Update the fitness score
        genome.fitness = total_reward
        
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
    

    for i in range(50): # Run for 50 generations
        winner = population.run(eval_genomes, 1) # Run the simulation for 1 generation and get the best pilot
        
        if i % 10 == 0 or i == 49: # Every 10 generations
            # Save that generation's brain profile: 'brain_gen_X.png'
            draw_net(config, winner, False, f"brains/brain_gen_{i}")
            
            with open(f'evolution_snapshots/pilot_gen_{i}.pkl', 'wb') as f:
                pickle.dump(winner, f)
            print(f"Saved snapshot of generation {i} pilot brain.")
        
            preview_winner(winner, config)
        
    # Save the champion
    print("Training complete! Saving the final champion...")
    with open(f'evolution_snapshots/pilot_gen_{i}.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print("Champion saved as 'best_pilot_brain.pkl'")
 

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
        screen = pygame.display.set_mode((1000, 600))
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
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 600))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 600), 3)
            draw_hud(env, reward, observation, 0, fuel_spent) # Draw the HUD
            draw_realtime_brain(winner_genome, config, observation) # Draw the brain activity
            pygame.display.flip() # refresh the screen
            
        print("Flight finished. Resetting in 2 seconds...")
        import time
        time.sleep(2) # Pause so you can see the final position

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
    screen = pygame.display.set_mode((1000, 600))
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
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 600))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 600), 3)

            draw_hud(env, total_reward, observation, step, fuel_spent=fuel_spent, gen_label=gen_label)
            draw_realtime_brain(genome, config, observation)
            
            pygame.display.flip()

        #print(f"Gen {gen_label} flight finished. Next pilot in 2 seconds...")
        #time.sleep(2)

    env.close()
    print("Evolution Tour Complete!")
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    
    #run_neat(config_path)
    #test_best_pilot(config_path, 'best_pilot_brain.pkl')
    test_evolution_history(config_path, 'evolution_snapshots')