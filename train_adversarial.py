import gymnasium as gym
import neat
import os
import pickle
import sys
import random
import copy
import itertools

# Import modularized tools
from evaluate import test_pilot, playback_evolution, validate_pilot, validate_pilot_precision
from visualize import visualize_all_checkpoints, plot_fitness_from_checkpoints, plot_smoothed_fitness

for folder in ['checkpoints_adversarial', 'adversarial_evolution_snapshots', 'adversarial_evolution_plots']:
    os.makedirs(folder, exist_ok=True)

SABOTEUR_PATH = 'saboteur_brain/budget_consc_saboteur.pkl'
PILOT_PATH = 'pilot_brain/best_pilot_brain.pkl'
CONFIG_PATH = 'config-feedforward'
current_generation = 0

def eval_genomes(genomes, config):
    global current_generation
    env = gym.make("LunarLander-v2")
    try:
        with open(SABOTEUR_PATH, 'rb') as f:
            saboteur_forces = pickle.load(f).forces 
    except FileNotFoundError:
        sys.exit("CRITICAL ERROR: Saboteur.pkl not found. Run Phase 2 first!")
        
    scaling_factor = min(1.0, 0.2 + (current_generation / 200.0))
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run_scores = []
        seeds = [42] + [random.randint(0, 100000) for _ in range(9)] 
        
        for i, sim_seed in enumerate(seeds):
            observation, info = env.reset(seed=sim_seed)
            total_run_reward = 0
            is_adversarial = (i < 8) 
            
            for step in range(600):
                # Apply Saboteur Force
                segment_index = min(step // 30, len(saboteur_forces) - 1)
                wind_force = saboteur_forces[segment_index] * scaling_factor
                env.unwrapped.lander.ApplyForceToCenter((wind_force, 0.0), True)
                
                output = net.activate(observation)
                action = output.index(max(output))
                observation, reward, terminated, truncated, info = env.step(action)

                x_pos, y_pos = observation[0], observation[1]
                v_horz, v_vert = observation[2], observation[3]
                angle = observation[4]
                left_leg, right_leg = observation[6], observation[7]
                dist_from_center = abs(x_pos)

                boundary_penalty = ((dist_from_center - 0.8) * 70.0) ** 2 if dist_from_center > 0.8 else 0
                center_reward = 1.5 - (dist_from_center ** 2) * 8.0
                
                if y_pos < 0.25:
                    stability_penalty = (abs(angle) * 10.0) + (abs(v_horz) * 8.0)
                    if v_vert < -0.10: stability_penalty += abs(v_vert) * 25.0 
                    if left_leg and right_leg:
                        if action != 0: total_run_reward -= 15.0 
                        else: total_run_reward += 25.0 
                    elif left_leg or right_leg:
                        if action != 0: total_run_reward -= 8.0
                else:
                    stability_penalty = (abs(angle) * 0.5) + (abs(v_horz) * 0.2)

                descent_pressure = 0
                if y_pos > 0.1:
                    if v_vert < -0.35: descent_pressure = -2.0
                    elif v_vert < -0.05: descent_pressure = 2.0
                    else: descent_pressure = -1.0

                total_run_reward += (reward + center_reward + descent_pressure - stability_penalty - boundary_penalty)
                total_run_reward -= 0.10

                if terminated or truncated:
                    if reward <= -100: total_run_reward -= 400 
                    if reward >= 100:
                        if dist_from_center < 0.05: total_run_reward += 5000.0 
                        elif dist_from_center < 0.15: total_run_reward += 2000.0
                        else: total_run_reward += 1000.0 
                    break
            run_scores.append(total_run_reward)
        genome.fitness = sum(run_scores) / len(run_scores)
    current_generation += 1    
    env.close()
 
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    population = neat.Population(config)
    
    if os.path.exists(PILOT_PATH):
        print("\n--- PHASE 3: Hardening the Veteran Pilot ---")
        with open(PILOT_PATH, 'rb') as f: veteran_genome = pickle.load(f)
        
        for g in population.population.values():
            g.nodes = copy.deepcopy(veteran_genome.nodes)
            g.connections = copy.deepcopy(veteran_genome.connections)
            g.fitness = None 
            
        max_node = max(veteran_genome.nodes.keys())
        innovation_numbers = [conn.innovation if hasattr(conn, 'innovation') else conn.key for conn in veteran_genome.connections.values() if isinstance(getattr(conn, 'innovation', getattr(conn, 'key', None)), int)]
        max_conn = max(innovation_numbers) if innovation_numbers else max_node + 1000 

        config.genome_config.node_indexer = itertools.count(max_node + 1)
        config.genome_config.connection_indexer = itertools.count(max_conn + 1)
        
        population.species.speciate(config, population.population, population.generation) 
    else:
        print("\n--- WARNING: No veteran found. Starting from scratch... ---")

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter() 
    population.add_reporter(stats)
    
    checkpoint_prefix = os.path.join('checkpoints_adversarial', 'adversarial-neat-checkpoint-')
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix))

    winner = population.run(eval_genomes, 500) 
        
    print("Training complete! Saving the Robust Champion...")
    with open('robust_final_pilot.pkl', 'wb') as f: pickle.dump(winner, f)
    best_ever = stats.best_genome() 
    print(f"Final Robust Fitness: {best_ever.fitness}")
    with open('robust_pilot_brain.pkl', 'wb') as f: pickle.dump(best_ever, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python train_adversarial.py [train/test_best/validate/validate_precision/plots/playback]")
        
    command = sys.argv[1].lower()
    
    if command == "train":
        if len(sys.argv) > 2:
            checkpoint_file = sys.argv[2]
            p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            p.config.fitness_threshold = 12000
            p.config.conn_add_prob = 0.1
            p.config.node_add_prob = 0.05
            p.config.max_stagnation = 20
            p.config.weight_mutate_power = 0.2
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join('checkpoints_adversarial', 'adversarial-neat-checkpoint-')))
            
            winner = p.run(eval_genomes, 50) 
            with open('robust_final_pilot.pkl', 'wb') as f: pickle.dump(winner, f)
            best_ever = stats.best_genome() 
            with open('robust_pilot_brain.pkl', 'wb') as f: pickle.dump(best_ever, f)
        else:
            run_neat(CONFIG_PATH)

    elif command == "test_best":
        if len(sys.argv) < 3: sys.exit("Please provide the genome file to test.")
        # We pass SABOTEUR_PATH here so the evaluation loop applies the wind
        test_pilot(CONFIG_PATH, sys.argv[2], saboteur_path=SABOTEUR_PATH)
        
    elif command == "validate":
        if len(sys.argv) < 3: sys.exit("Usage: python train_adversarial.py validate [genome_file]")
        validate_pilot(CONFIG_PATH, sys.argv[2])
        
    elif command == "validate_precision":
        if len(sys.argv) < 3: sys.exit("Usage: python train_adversarial.py validate_precision [genome_file]")
        validate_pilot_precision(CONFIG_PATH, sys.argv[2])
        
    elif command == "plots":
        os.makedirs("adversarial_evolution_plots", exist_ok=True)
        visualize_all_checkpoints("checkpoints_adversarial", CONFIG_PATH)
        plot_fitness_from_checkpoints('checkpoints_adversarial', filename='adversarial_evolution_graph.png')
        plot_smoothed_fitness('checkpoints_adversarial', window_size=5, filename='adversarial_smoothed_evolution.png')
        
    elif command == "playback":
        playback_evolution('checkpoints_adversarial', CONFIG_PATH, interval=10)
        
    else:
        print("Unknown command.")