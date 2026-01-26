import gymnasium as gym
import pickle
import neat
import os

def record_pilot(genome_path, config_path, seed=500):
    # 1. Load the Brain
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # 2. Load the Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # 3. Create the Network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # 4. Setup Environment with Video Recording
    # This will save the video into a folder called 'victory_lap'
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="victory_lap", 
                                   name_prefix=f"elite_pilot_seed_{seed}")

    observation, info = env.reset(seed=seed)
    total_reward = 0
    
    print(f"--- Recording Victory Lap on Seed {seed} ---")

    for _ in range(1000):
        # Activate the brain (the nodes and connections from your image!)
        output = net.activate(observation)
        action = output.index(max(output))
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Trial Finished! Total Reward: {total_reward:.2f}")
    env.close()
    print(f"Video saved to: {os.getcwd()}/victory_lap")

def create_highlight_reel(genome_path, config_path, seeds=[42, 101, 500, 777, 1337]):
    # 1. Load the Brain
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # 2. Load the Config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # 3. Create the Network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Ensure output directory exists
    video_dir = "highlight_reel"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    for seed in seeds:
        # 4. Setup Environment with Video Recording
        # Each seed gets its own subfolder to avoid overwriting
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=os.path.join(video_dir, f"seed_{seed}"), 
            name_prefix="elite_performance",
            disable_logger=True
        )

        observation, info = env.reset(seed=seed)
        total_reward = 0
        
        print(f"--- Recording Highlight: Seed {seed} ---")

        for _ in range(1000):
            output = net.activate(observation)
            action = output.index(max(output))
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Seed {seed} Complete! Final Score: {total_reward:.2f}")
        env.close()

    print(f"\nAll highlights saved to the '{video_dir}' directory.")
    
if __name__ == "__main__":
    # Point these to your specific files
    #record_pilot("best_pilot_brain.pkl", "config-feedforward")
    create_highlight_reel("best_pilot_brain.pkl", "config-feedforward")