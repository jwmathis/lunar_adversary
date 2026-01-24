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
        
        # Save that generation's brain profile: 'brain_gen_X.png'
        draw_net(config, winner, False, f"brains/brain_gen_{i}")
        
        print(f"Generation {i} best fitness: {winner.fitness}")
        #preview_winner(winner, config)
        
    # Save the champion
    print("Training complete! Saving the final champion...")
    with open('best_pilot_brain.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print("Champion saved as 'best_pilot_brain.pkl'")
 
 
"""
Description: Preview the best brain found in a generation

Parameters
----------
winner : neat.DefaultGenome
    The winner genome
config : neat.config.Config
    The configuration file
"""
def preview_winner(winner, config):
    # Create a visual environment for the winner's victory lap
    visual_env = gym.make("LunarLander-v3", render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Reset the environment
    observation, info = visual_env.reset()
    fuel_spent = 0.0
    for step in range(1000):
        visual_env.render()  # Render the environment
        outputs = net.activate(observation)
        action = outputs.index(max(outputs)) # choose the highest signal
        
        if action == 2: # Main engine
            fuel_spent += 0.3
        elif action in [1, 3]: # Side engines
            fuel_spent += 0.03
             
        observation, reward, terminated, truncated, info = visual_env.step(action)
        
        draw_hud(visual_env, reward, observation, step, fuel_spent=fuel_spent) # Draw the HUD
        pygame.display.flip() # refresh the screen
        
        if terminated or truncated:
            time.sleep(2)
            break

    visual_env.close()
    print("Preview complete")
    #input("Press Enter to continue...")

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
            draw_hud(env, reward, observation, 0, fuel_spent) # Draw the HUD
            pygame.display.flip() # refresh the screen
            
        print("Flight finished. Resetting in 2 seconds...")
        import time
        time.sleep(2) # Pause so you can see the final position

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    
    #run_neat(config_path)
    test_best_pilot(config_path, 'best_pilot_brain.pkl')