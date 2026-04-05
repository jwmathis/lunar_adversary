import graphviz
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import neat
import os

def draw_net(config, genome, view=False, filename="best_pilot_brain"):
    """Draw the neural network of the best pilot"""
    dot = graphviz.Digraph(format='png', engine='dot')
    dot.attr(bgcolor='#2E2E2E')
    dot.attr('node', fontname='Helvetica', fontsize='10', fontcolor='white', style='filled', fillcolor='#4B4B4B', color='#4B4B4B')
    
    for node_id in config.genome_config.input_keys:
        dot.node(str(node_id), label=f"Sensor {node_id}", shape='hexagon', color='white', style='filled', fillcolor='#4A90E2')
    for node_id in config.genome_config.output_keys:
        dot.node(str(node_id), label=f"Engine {node_id}", shape='circle', color='white', style='filled', fillcolor='#50E3C2')
    
    for cg in genome.connections.values():
        if cg.enabled:
            input_node, output_node = cg.key
            width = str(0.5 + abs(cg.weight) * 0.8)
            color = '#7ED321' if cg.weight > 0 else '#D0021B'
            dot.edge(str(input_node), str(output_node), penwidth=width, arrowhead='none', color=color)
            
    dot.render(filename, view=view, cleanup=True)

def draw_realtime_brain(genome, config, current_observations):
    """Visualize the real-time brain activity of the pilot"""
    canvas = pygame.display.get_surface()
    if canvas is None: return 
    
    font = pygame.font.SysFont('Arial', 14, bold=True)
    input_labels = ["Pos X", "Pos Y", "H-Vel", "V-Vel", "Angle", "Ang-Vel", "L-Leg", "R-Leg"]
    output_labels = ["Idle", "Left", "Main", "Right"]
    
    start_x, start_y = 720, 150
    layer_width = 200
    node_spacing = 45
    
    input_coords = {key: (start_x, start_y + (i * node_spacing)) for i, key in enumerate(config.genome_config.input_keys)}
    output_coords = {key: (start_x + layer_width, start_y + (i * (node_spacing * 2))) for i, key in enumerate(config.genome_config.output_keys)}

    hidden_nodes = [n for n in genome.nodes.keys() if n not in config.genome_config.output_keys]
    hidden_coords = {node_id: (start_x + (layer_width // 2), start_y + (i * 50)) for i, node_id in enumerate(hidden_nodes)}

    all_coords = {**input_coords, **output_coords, **hidden_coords}
    
    # Draw Connections
    for conn in genome.connections.values():
        if not conn.enabled: continue
        in_node, out_node = conn.key
        if in_node in all_coords and out_node in all_coords:
            p1 = all_coords[in_node]
            p2 = all_coords[out_node]
            color = (0, 255, 0) if conn.weight > 0 else (255, 0, 0)
            thickness = max(1, int(abs(conn.weight) * 2))
            pygame.draw.line(canvas, color, p1, p2, thickness)

    # Draw Nodes
    for i, (key, coords) in enumerate(input_coords.items()):
        val = abs(current_observations[i])
        color = (255, 255, 255) if val > 0.1 else (100, 100, 100)
        node_color = (100, 100, 255) if val > 0.1 else (50, 50, 150)
        canvas.blit(font.render(input_labels[i], True, color), (coords[0] - 75, coords[1] - 7)) 
        pygame.draw.circle(canvas, node_color, coords, 10)

    for i, (key, coords) in enumerate(output_coords.items()):
        canvas.blit(font.render(output_labels[i], True, (200, 200, 200)), (coords[0] + 20, coords[1] - 7)) 
        pygame.draw.circle(canvas, (255, 100, 100), coords, 10)

    for coords in hidden_coords.values():
        pygame.draw.circle(canvas, (180, 180, 180), coords, 8)

def draw_hud(env, total_reward, observation, step, fuel_spent, wind_force=0.0, gen_label="Unknown Gen"):
    """Draw a heads-up display (HUD) on the Lunar Lander screen"""
    canvas = env.unwrapped.screen
    if canvas is None: return 
    
    font = pygame.font.SysFont('Arial', 20, bold=True)
    left_leg = "YES" if observation[6] == 1.0 else "NO"
    right_leg = "YES" if observation[7] == 1.0 else "NO"
    
    stats = [
        f"GENERATION: {gen_label}", f"Step: {step}", f"Score: {total_reward:.2f}",
        f"V-Speed: {observation[3]:.2f}", f"H-Speed: {observation[2]:.2f}",
        f"Angle: {observation[4]:.2f}", f"Fuel Spent: --{fuel_spent:.1f}",
        f"L-Leg: {left_leg} | R-Leg: {right_leg}", f"Wind force: {wind_force:.2f}"
    ]
    
    overlay = pygame.Surface((180, 140))
    overlay.set_alpha(160)
    overlay.fill((0, 0, 0))
    canvas.blit(overlay, (10, 10))    
    
    for i, text in enumerate(stats):
        color = (255, 255, 255)
        if "Score" in text:
            color = (0, 255, 0) if total_reward > 0 else (255, 50, 50)
        canvas.blit(font.render(text, True, color), (20, 20 + i * 20))

def visualize_checkpoint_brain(checkpoint_path, config_path):
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    draw_net(config, p.best_genome, False, filename=f"brain_from_{checkpoint_path}")
    print(f"Brain diagram saved for {checkpoint_path}")
    
def visualize_all_checkpoints(folder_path, config_path):
    files = [f for f in os.listdir(folder_path) if f.startswith('neat-checkpoint-') or f.startswith('adversarial-neat-checkpoint-')]
    files.sort(key=lambda x: int(x.split('-')[-1]))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    for filename in files:
        p = neat.Checkpointer.restore_checkpoint(os.path.join(folder_path, filename))
        best_genome = max((g for g in p.population.values() if g.fitness is not None), key=lambda g: g.fitness, default=None)
        if best_genome is None: continue
        
        gen_num = filename.split('-')[-1]
        try:
            draw_net(config, best_genome, False, filename=f"evolution_plots/brain_gen_{gen_num}")
        except Exception as e:
            print(f"Error drawing generation {gen_num}: {e}")

def plot_fitness_from_checkpoints(checkpoint_folder, filename="evolution_graph.png"):
    files = [f for f in os.listdir(checkpoint_folder) if 'checkpoint-' in f]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    generations, best_fitness, avg_fitness = [], [], []

    for f_name in files:
        p = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_folder, f_name))
        all_fitnesses = [g.fitness for g in p.population.values() if g.fitness is not None]
        if all_fitnesses:
            generations.append(int(f_name.split('-')[-1]))
            best_fitness.append(max(all_fitnesses))
            avg_fitness.append(sum(all_fitnesses) / len(all_fitnesses))

    plt.style.use('dark_background') 
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, label='Best Pilot', color='#50E3C2', linewidth=2)
    plt.plot(generations, avg_fitness, label='Average Population', color='#4A90E2', linestyle='--')
    plt.title('Pilot Intelligence Evolution (Reconstructed from Checkpoints)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Score)')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
 
def plot_smoothed_fitness(checkpoint_folder, window_size=5, filename="smoothed_evolution.png"):
    files = [f for f in os.listdir(checkpoint_folder) if 'checkpoint-' in f]
    files.sort(key=lambda x: int(x.split('-')[-1]))

    generations, best_fitness, avg_fitness = [], [], []

    for f_name in files:
        p = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_folder, f_name))
        all_fits = [g.fitness for g in p.population.values() if g.fitness is not None]
        if all_fits:
            generations.append(int(f_name.split('-')[-1]))
            best_fitness.append(max(all_fits))
            avg_fitness.append(sum(all_fits) / len(all_fits))

    smoothed_best = pd.Series(best_fitness).rolling(window=window_size, min_periods=1).mean()
    smoothed_avg = pd.Series(avg_fitness).rolling(window=window_size, min_periods=1).mean()

    plt.style.use('dark_background') 
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, color='#50E3C2', alpha=0.2, label='_nolegend_')
    plt.plot(generations, avg_fitness, color='#4A90E2', alpha=0.1, label='_nolegend_')
    plt.plot(generations, smoothed_best, label=f'Best ({window_size}-Gen Avg)', color='#50E3C2', linewidth=3)
    plt.plot(generations, smoothed_avg, label=f'Pop. Mean ({window_size}-Gen Avg)', color='#4A90E2', linewidth=2, linestyle='--')
    
    plt.title('Smoothed Pilot Intelligence Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(color='gray', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)

def plot_precision_histogram(displacements, filename="landing_precision.png"):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.hist(displacements, bins=20, color='#50E3C2', alpha=0.7, edgecolor='white')
    plt.axvspan(0, 0.2, color='green', alpha=0.1, label='Inside Flags (Success Zone)')
    plt.axvline(0.2, color='red', linestyle='--', alpha=0.5, label='Flag Boundary')
    
    plt.title('Landing Precision Distribution (Distance from Center)')
    plt.xlabel('Offset from Center (0.0 = Perfect Bullseye)')
    plt.ylabel('Number of Landings')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(filename)