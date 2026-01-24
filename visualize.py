import graphviz
import pygame

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
    
    # Define Nodes
    for node_id in config.genome_config.input_keys:
        dot.node(str(node_id), shape='box', color='lightblue', style='filled')
    for node_id in config.genome_config.output_keys:
        dot.node(str(node_id), shape='circle', color='lightgreen', style='filled')
    
    # Define Connections
    for cg in genome.connections.values():
        if cg.enabled:
            input_node, output_node = cg.key
            weight = f"{cg.weight:.2f}"
            color = 'green' if cg.weight > 0 else 'red'
            dot.edge(str(input_node), str(output_node), label=weight, color=color)
            
    dot.render(filename, view=view, cleanup=True)


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
def draw_hud(env, total_reward, observation, step, fuel_spent, wind_force=0.0):
    # Access the pygame surface from the Gymnasium environment
    canvas = env.unwrapped.screen
    if canvas is None: return # Safety check
    
    font = pygame.font.SysFont('Arial', 20, bold=True)
    left_leg = "YES" if observation[6] == 1.0 else "NO"
    right_leg = "YES" if observation[7] == 1.0 else "NO"
    
    # The data strings
    stats = [
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
