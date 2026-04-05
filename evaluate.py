import gymnasium as gym
import neat
import os
import pickle
import time
import pygame
import numpy as np
from visualize import draw_hud, draw_realtime_brain, plot_precision_histogram

def test_pilot(config_path, genome_path, saboteur_path=None):
    """Tests a pilot. If saboteur_path is provided, applies adversarial forces."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    with open(genome_path, 'rb') as f:
        winner_genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    
    saboteur = None
    if saboteur_path and os.path.exists(saboteur_path):
        with open(saboteur_path, 'rb') as f:
            saboteur = pickle.load(f)

    env = gym.make("LunarLander-v2", render_mode="human")

    while True: 
        observation, info = env.reset()
        screen = pygame.display.set_mode((1000, 1000))
        terminated = truncated = False
        fuel_spent = 0.0
        step = 0
        
        print("Pilot is starting a new flight...")

        while not (terminated or truncated):
            wind_force = 0.0
            
            # Apply Saboteur Force if evaluating in adversarial mode
            if saboteur:
                segment_index = min(step // 30, len(saboteur.forces) - 1)
                wind_force = saboteur.forces[segment_index]
                env.unwrapped.lander.ApplyForceToCenter((float(wind_force), 0.0), True)

            outputs = net.activate(observation)
            action = outputs.index(max(outputs))
            if action == 2: fuel_spent += 0.3
            elif action in [1, 3]: fuel_spent += 0.03
                
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Visuals
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 1000))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 1000), 3)
            draw_hud(env, reward, observation, step, fuel_spent, wind_force)
            draw_realtime_brain(winner_genome, config, observation)
            
            # Draw Saboteur Force Arrow if active
            if saboteur:
                start_pos = (400, 100)
                end_pos = (400 + int(wind_force * 5), 100)
                if abs(wind_force) > 15: color = (255, 50, 50)
                elif abs(wind_force) > 5: color = (200, 0, 255)
                else: color = (200, 200, 200)
                pygame.draw.line(screen, color, start_pos, end_pos, 5)
                pygame.draw.circle(screen, color, end_pos, 7)
            
            pygame.display.flip()
            step += 1
            
        print("Flight finished. Resetting in 2 seconds...")
        time.sleep(2)

def playback_evolution(checkpoint_folder, config_path, interval=50):
    env = gym.make("LunarLander-v2", render_mode="human")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    files = [f for f in os.listdir(checkpoint_folder) if 'checkpoint-' in f]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    for filename in files:
        gen_num = int(filename.split('-')[-1])
        if gen_num % interval != 0 and gen_num != int(files[-1].split('-')[-1]): 
            continue

        print(f"\n--- Showing Pilot from Generation {gen_num} ---")
        p = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_folder, filename))
        best_genome = max((g for g in p.population.values() if g.fitness is not None), key=lambda g: g.fitness, default=None)
        if best_genome is None: continue

        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        observation, info = env.reset(seed=42) 
        screen = pygame.display.set_mode((1000, 1000))
        done = False
        total_reward = fuel_spent = 0.0
        step = 0
        
        while not done:
            env.render()
            output = net.activate(observation)
            action = output.index(max(output))
            if action == 2: fuel_spent += 0.3
            elif action in [1, 3]: fuel_spent += 0.03
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 1000))
            pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 1000), 3)
            draw_hud(env, total_reward, observation, step, fuel_spent, gen_label=gen_num)
            draw_realtime_brain(best_genome, config, observation)
            pygame.display.flip()
            step += 1
            
        print(f"Generation {gen_num} Result: {total_reward:.2f}")
        time.sleep(1)

    env.close()

def validate_pilot(config_path, genome_path, num_episodes=50):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    with open(genome_path, 'rb') as f:
        winner_genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    env = gym.make("LunarLander-v2") 

    successes = crashes = total_reward = 0

    print(f"\n--- Validating Pilot over {num_episodes} Random Maps ---")
    for i in range(num_episodes):
        observation, info = env.reset()
        terminated = truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            outputs = net.activate(observation)
            action = outputs.index(max(outputs))
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        if episode_reward >= 100: successes += 1
        elif episode_reward < -100: crashes += 1
        total_reward += episode_reward
        print(f"Trial {i+1}: Reward = {episode_reward:.2f}")

    print("\n--- FINAL VALIDATION RESULTS ---")
    print(f"Success Rate: {(successes/num_episodes)*100:.1f}%")
    print(f"Crash Rate: {(crashes/num_episodes)*100:.1f}%")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    env.close()

def validate_pilot_precision(config_path, genome_path, num_trials=50):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make("LunarLander-v2")
    
    rewards, displacements = [], []
    success_count = 0

    print(f"--- Validating Precision over {num_trials} Random Maps ---")
    for i in range(num_trials):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            output = net.activate(observation)
            action = np.argmax(output)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if done:
                final_x = observation[0]
                displacements.append(abs(final_x))
                if episode_reward >= 200: success_count += 1
        
        rewards.append(episode_reward)
        print(f"Trial {i+1}: Reward = {episode_reward:>7.2f} | Final X = {final_x:>6.3f}")

    avg_reward = sum(rewards) / num_trials
    avg_precision = sum(displacements) / num_trials
    success_rate = (success_count / num_trials) * 100
    
    print("\n" + "="*40 + "\n      FINAL PRECISION RESULTS\n" + "="*40)
    print(f"Success Rate:     {success_rate:.1f}%")
    print(f"Average Reward:   {avg_reward:.2f}")
    print(f"Average Offset:   {avg_precision:.4f} units")
    if avg_precision < 0.05: print("Rating: SNIPER (Inside the bullseye)")
    elif avg_precision < 0.2: print("Rating: PROFESSIONAL (Inside the flags)")
    else: print("Rating: ROOKIE (Safe but scattered)")
    print("="*40)
    env.close()
    plot_precision_histogram(displacements)

def record_pilot(config_path, genome_path, seed=500):
    """Records a video of the pilot's performance."""
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="victory_lap", name_prefix=f"elite_pilot_seed_{seed}")

    observation, info = env.reset(seed=seed)
    total_reward = 0
    print(f"--- Recording Victory Lap on Seed {seed} ---")

    for _ in range(1000):
        output = net.activate(observation)
        action = output.index(max(output))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated: break

    print(f"Trial Finished! Total Reward: {total_reward:.2f}")
    env.close()
    print(f"Video saved to: {os.getcwd()}/victory_lap")