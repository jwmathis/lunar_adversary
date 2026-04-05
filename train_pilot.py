import gymnasium as gym
import neat
import os
import pickle
import random
import argparse

# Import modularized tools
from evaluate import test_pilot, playback_evolution, validate_pilot, validate_pilot_precision, record_pilot
from visualize import visualize_all_checkpoints, plot_fitness_from_checkpoints, plot_smoothed_fitness, visualize_checkpoint_brain

for folder in ['checkpoints', 'evolution_snapshots', 'evolution_plots']:
    os.makedirs(folder, exist_ok=True)
        
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
                    if reward <= -100: total_run_reward -= 800 
                    if reward >= 100:
                        if dist_from_center < 0.05: total_run_reward += 1000.0 
                        elif dist_from_center < 0.15: total_run_reward += 500.0 
                    break
            
            run_scores.append(total_run_reward)
        genome.fitness = sum(run_scores) / len(run_scores)
    env.close()
 
def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    population = neat.Population(config) 
    
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter() 
    population.add_reporter(stats) 
    
    checkpoint_prefix = os.path.join('checkpoints', 'neat-checkpoint-')
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix)) 

    winner = population.run(eval_genomes, 500) 
        
    print("Training complete! Saving the final champion...")
    with open('final_pilot_brain.pkl', 'wb') as f: pickle.dump(winner, f)
        
    best_ever = stats.best_genome() 
    print(f"Final Best Fitness: {best_ever.fitness}")
    with open('best_pilot_brain.pkl', 'wb') as f: pickle.dump(best_ever, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(local_dir, "config-feedforward")
    
    parser = argparse.ArgumentParser(description="Phase 1: Baseline NEAT Pilot Training")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    # 1. TRAIN
    train_parser = subparsers.add_parser("train", help="Train a pilot from scratch")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint file to resume from")
    
    # 2. EVALUATE
    eval_parser = subparsers.add_parser("evaluate", help="Test, validate, or record a pilot")
    eval_parser.add_argument("mode", choices=["test_best", "validate", "validate_precision", "record"], help="Evaluation mode")
    eval_parser.add_argument("pilot_path", type=str, help="Path to the pilot .pkl brain")
    
    # 3. VISUALIZE
    viz_parser = subparsers.add_parser("visualize", help="Generate plots, playback, or brain diagrams")
    viz_parser.add_argument("mode", choices=["plots", "playback", "draw_brain"], help="Visualization mode")
    viz_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint (required for draw_brain)")

    args = parser.parse_args()
    
    if args.command == "train":
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            p = neat.Checkpointer.restore_checkpoint(args.resume)
            p.config.fitness_threshold = 12000
            p.config.conn_add_prob = 0.1
            p.config.node_add_prob = 0.05
            p.config.max_stagnation = 20
            p.config.weight_mutate_power = 0.2
            
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join('checkpoints', 'neat-checkpoint-')))
            
            winner = p.run(eval_genomes, 50) 
            with open('final_pilot_brain.pkl', 'wb') as f: pickle.dump(winner, f)
            best_ever = stats.best_genome() 
            with open('best_pilot_brain.pkl', 'wb') as f: pickle.dump(best_ever, f)
        else:
            run_neat(CONFIG_PATH)

    elif args.command == "evaluate":
        if args.mode == "test_best":
            test_pilot(CONFIG_PATH, args.pilot_path)
        elif args.mode == "validate":
            validate_pilot(CONFIG_PATH, args.pilot_path)
        elif args.mode == "validate_precision":
            validate_pilot_precision(CONFIG_PATH, args.pilot_path)
        elif args.mode == "record":
            record_pilot(CONFIG_PATH, args.pilot_path)
            
    elif args.command == "visualize":
        if args.mode == "plots":
            os.makedirs("evolution_plots", exist_ok=True)
            visualize_all_checkpoints("checkpoints", CONFIG_PATH)
            plot_fitness_from_checkpoints('checkpoints')
            plot_smoothed_fitness('checkpoints', window_size=5)
        elif args.mode == "playback":
            playback_evolution('checkpoints', CONFIG_PATH, interval=10)
        elif args.mode == "draw_brain":
            if not args.checkpoint:
                parser.error("draw_brain requires the --checkpoint argument")
            visualize_checkpoint_brain(args.checkpoint, CONFIG_PATH)