import graphviz
import matplotlib.pyplot as plt
import pygame
import neat
import gymnasium as gym
import time
import os

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
    screen = pygame.display.set_mode((1000, 600))
    fuel_spent = 0.0
    total_reward = 0.0
    
    for step in range(1000):
        visual_env.render()  # Render the environment
        outputs = net.activate(observation)
        action = outputs.index(max(outputs)) # choose the highest signal
        
        if action == 2: # Main engine
            fuel_spent += 0.3
        elif action in [1, 3]: # Side engines
            fuel_spent += 0.03
             
        observation, reward, terminated, truncated, info = visual_env.step(action)
        total_reward += reward
        
        pygame.draw.rect(screen, (30, 30, 30), (600, 0, 400, 600))
        pygame.draw.line(screen, (200, 200, 200), (600, 0), (600, 600), 3)
        draw_hud(visual_env, reward, observation, step, fuel_spent=fuel_spent) # Draw the HUD
        draw_realtime_brain( winner, config, observation) # Draw the brain activity
        pygame.display.flip() # refresh the screen
        
        if terminated or truncated:
            time.sleep(2)
            break

    visual_env.close()
    print("Preview complete")
    #input("Press Enter to continue...")
    
"""
Description: Draw the neural network of the best pilot

Parameters 
----------
config : neat.config.Config
    The configuration file
genome : neat.DefaultGenome
    The genome of the best pilot
view : bool, optional
    Whether to view the graph, by default False
filename : str, optional
    The filename of the graph, by default "best_pilot_brain
"""
def draw_net(config, genome, view=False, filename="best_pilot_brain"):
    dot = graphviz.Digraph(format='png', engine='dot')
    
    dot.attr(bgcolor='#2E2E2E')  # Dark background
    dot.attr('node', fontname='Helvetica', fontsize='10', fontcolor='white', style='filled', fillcolor='#4B4B4B', color='#4B4B4B')
    
    # Define Input Nodes (Sensors)
    for node_id in config.genome_config.input_keys:
        dot.node(str(node_id), label=f"Sensor {node_id}", shape='hexagon', color='white', style='filled', fillcolor='#4A90E2')
    
    # Define Output Nodes (Engines)
    for node_id in config.genome_config.output_keys:
        dot.node(str(node_id), label=f"Engine {node_id}", shape='circle', color='white', style='filled', fillcolor='#50E3C2')
    
    # Define Connections
    for cg in genome.connections.values():
        if cg.enabled:
            input_node, output_node = cg.key
            width = str(0.5 + abs(cg.weight) * 0.8)  # Scale width for visibility
            color = '#7ED321' if cg.weight > 0 else '#D0021B'  # Green for positive, red for negative
            dot.edge(str(input_node), str(output_node), penwidth=width, arrowhead='none', color=color)
            
    dot.render(filename, view=view, cleanup=True)

"""
Description: Visualize the real-time brain activity of the pilot

Parameters
----------
env : gymnasium.core.Env
    The environment
genome : neat.DefaultGenome
    The genome of the best pilot
config : neat.config.Config
    The configuration file
current_observations : list
    The current observations of the lander
"""
def draw_realtime_brain(genome, config, current_observations):
    canvas = pygame.display.get_surface()
    if canvas is None: return # Safety check
    
    font = pygame.font.SysFont('Arial', 14, bold=True)
    
    # Label Lists
    input_labels = ["Pos X", "Pos Y", "H-Vel", "V-Vel", "Angle", "Ang-Vel", "L-Leg", "R-Leg"]
    output_labels = ["Idle", "Left", "Main", "Right"]
    
    # Define the area for the brain (Top-right corner)
    start_x, start_y = 720, 150
    layer_width = 200
    node_spacing = 45
    
    # 1. Map coordinates for Nodes
    # Map Input Nodes
    input_coords = {key: (start_x, start_y + (i * node_spacing)) 
                    for i, key in enumerate(config.genome_config.input_keys)}
    # Map Output Nodes
    output_coords = {key: (start_x + layer_width, start_y + (i * (node_spacing * 2))) 
                     for i, key in enumerate(config.genome_config.output_keys)}

    # Map Hidden Nodes
    hidden_nodes = [n for n in genome.nodes.keys() if n not in config.genome_config.output_keys]
    hidden_coords = {}
    for i, node_id in enumerate(hidden_nodes):
        # Place them in the middle (x) and spread them out (y)
        hidden_coords[node_id] = (start_x + (layer_width // 2), start_y + (i * 50))

    for i, node_id in enumerate(hidden_nodes):
        hidden_coords[node_id] = (start_x + (layer_width // 2), start_y + (i * 50))

    # 2. Draw Connections FIRST (so they stay behind the nodes)
    all_coords = {**input_coords, **output_coords, **hidden_coords}
    for conn in genome.connections.values():
        if not conn.enabled: continue
        in_node, out_node = conn.key
        if in_node in all_coords and out_node in all_coords:
            p1 = all_coords[in_node]
            p2 = all_coords[out_node]
            color = (0, 255, 0) if conn.weight > 0 else (255, 0, 0)
            thickness = max(1, int(abs(conn.weight) * 2))
            pygame.draw.line(canvas, color, p1, p2, thickness)

    # 3. Draw Nodes and Dynamic Labels
    # Inputs
    for i, (key, coords) in enumerate(input_coords.items()):
        val = abs(current_observations[i])
        # Glow text and node if sensor is active
        color = (255, 255, 255) if val > 0.1 else (100, 100, 100)
        node_color = (100, 100, 255) if val > 0.1 else (50, 50, 150)
        
        label_surf = font.render(input_labels[i], True, color)
        canvas.blit(label_surf, (coords[0] - 75, coords[1] - 7)) # Label to the left
        pygame.draw.circle(canvas, node_color, coords, 10)

    # Outputs
    for i, (key, coords) in enumerate(output_coords.items()):
        label_surf = font.render(output_labels[i], True, (200, 200, 200))
        canvas.blit(label_surf, (coords[0] + 20, coords[1] - 7)) # Label to the right
        pygame.draw.circle(canvas, (255, 100, 100), coords, 10)

    # Hidden Nodes (No labels needed for these usually)
    for coords in hidden_coords.values():
        pygame.draw.circle(canvas, (180, 180, 180), coords, 8)
        
    # 2. Draw Connections
    all_coords = {**input_coords, **output_coords, **hidden_coords}
            
    # 2. Draw Connections
    for conn in genome.connections.values():
        if not conn.enabled: continue
        in_node, out_node = conn.key
        
        # Now check if BOTH ends of the connection exist in the big map
        if in_node in all_coords and out_node in all_coords:
            p1 = all_coords[in_node]
            p2 = all_coords[out_node]
            
            color = (0, 255, 0) if conn.weight > 0 else (255, 0, 0)
            thickness = max(1, int(abs(conn.weight) * 2))
            pygame.draw.line(canvas, color, p1, p2, thickness)
            
"""
Description: Plot the fitness statistics over generations

Parameters
----------
statistics : neat.StatisticsReporter
    The statistics reporter from the NEAT simulation
"""
def plot_stats(statistics, filename="fitness_over_time.png"):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_mean()
    
    plt.style.use('dark_background') 
    plt.figure(figsize=(10, 5))
    plt.plot(generation, best_fitness, label='Best Pilot', color='#50E3C2', linewidth=2)
    plt.plot(generation, avg_fitness, label='Average Population', color='#4A90E2', linestyle='--')
    
    plt.title('Pilot Intelligence Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Score)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig('evolution_graph.png')
    plt.show()

"""
Description: Draw a heads-up display (HUD) on the Lunar Lander screen

Parameters
----------
env : gymnasium.core.Env
    The Gymnasium environment
total_reward : float
    The total reward of the pilot so far
observation : list
    The current state of the lander
step : int
    The current step of the simulation
fuel_spent : float
    The amount of fuel spent so far
wind_force : float, optional
    The current wind force, by default 0.0
"""
def draw_hud(env, total_reward, observation, step, fuel_spent, wind_force=0.0, gen_label="Unknown Gen"):
    # Access the pygame surface from the Gymnasium environment
    canvas = env.unwrapped.screen
    if canvas is None: return # Safety check
    
    font = pygame.font.SysFont('Arial', 20, bold=True)
    left_leg = "YES" if observation[6] == 1.0 else "NO"
    right_leg = "YES" if observation[7] == 1.0 else "NO"
    
    # The data strings
    stats = [
        f"GENERATION: {gen_label}",
        f"Step: {step}",
        f"Score: {total_reward:.2f}",
        f"V-Speed: {observation[3]:.2f}",
        f"H-Speed: {observation[2]:.2f}",
        f"Angle: {observation[4]:.2f}",
        f"Fuel Spent: --{fuel_spent:.1f}",
        f"L-Leg: {left_leg} | R-Leg: {right_leg}",
        f"Wind force: {wind_force:.2f}"
    ]
    
    # Draw a simple HUD
    overlay = pygame.Surface((180, 140))
    overlay.set_alpha(160)
    overlay.fill((0, 0, 0))
    canvas.blit(overlay, (10, 10))    
    
    # Draw each line of text
    for i, text in enumerate(stats):
        color = (255, 255, 255)
        # Turn score red if it's failing, green if it's succeeding
        if "Score" in text:
            color = (0, 255, 0) if total_reward > 0 else (255, 50, 50)
            
            
        text_surface = font.render(text, True, color)
        canvas.blit(text_surface, (20, 20 + i * 20))

def visualize_checkpoint_brain(checkpoint_path, config_path):
    # 1. Load the checkpoint
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    
    # 2. Get the best genome from that checkpoint
    # You can also use p.population.values() to iterate through EVERY brain
    best_genome = p.best_genome
    
    # 3. Load the config (required for the drawing logic)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # 4. Draw the brain
    # This uses your visualize.py logic
    draw_net(config, best_genome, False, filename=f"brain_from_{checkpoint_path}")
    print(f"Brain diagram saved for {checkpoint_path}")
    
def visualize_all_checkpoints(folder_path, config_path):
    # Get all files that start with 'neat-checkpoint-'
    files = [f for f in os.listdir(folder_path) if f.startswith('neat-checkpoint-')]
    
    # Sort them numerically so the evolution looks chronological
    files.sort(key=lambda x: int(x.split('-')[-1]))

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    for filename in files:
        checkpoint_path = os.path.join(folder_path, filename)
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        
        # NEW LOGIC: Manually find the best genome in the population
        best_genome = None
        for g in p.population.values():
            if best_genome is None or (g.fitness is not None and g.fitness > best_genome.fitness):
                best_genome = g
        
        # Safety check: if for some reason we still don't have one, skip it
        if best_genome is None:
            print(f"Skipping {filename}: No genomes found.")
            continue

        gen_num = filename.split('-')[-1]
        
        # Ensure the visualize function gets what it needs
        try:
            draw_net(config, best_genome, False, filename=f"evolution_plots/brain_gen_{gen_num}")
            print(f"Generated brain diagram for Generation {gen_num}")
        except Exception as e:
            print(f"Error drawing generation {gen_num}: {e}")

    print("Done!")