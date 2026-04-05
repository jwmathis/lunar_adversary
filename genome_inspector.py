import pickle
import neat
import os
import argparse

# Since Saboteur is a custom class, we define a dummy version 
# to allow pickle to load the data if the original script isn't present.
try:
    from saboteur import Saboteur
except ImportError:
    class Saboteur:
        def __init__(self, num_segments=20):
            self.forces = []
            self.fitness = 0

def inspect_pilot(filepath, config_path='config-feedforward'):
    """Extracts NEAT metrics from a Pilot .pkl file."""
    if not os.path.exists(filepath):
        return f"Error: {filepath} not found."
    
    try:
        with open(filepath, 'rb') as f:
            genome = pickle.load(f)
        
        # Count active nodes and connections
        node_count = len(genome.nodes)
        conn_count = len([c for c in genome.connections.values() if c.enabled])
        fitness = genome.fitness
        
        print("-" * 30)
        print(f"PILOT ANALYSIS: {os.path.basename(filepath)}")
        print("-" * 30)
        print(f"Fitness Score:    {fitness}")
        print(f"Neural Nodes:     {node_count}")
        print(f"Enabled Conns:    {conn_count}")
        print(f"Genome ID:        {genome.key}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Could not parse Pilot file: {e}")

def inspect_saboteur(filepath):
    """Extracts metrics from a Saboteur .pkl file."""
    if not os.path.exists(filepath):
        return f"Error: {filepath} not found."
    
    try:
        with open(filepath, 'rb') as f:
            saboteur = pickle.load(f)
        
        print("-" * 30)
        print(f"SABOTEUR ANALYSIS: {os.path.basename(filepath)}")
        print("-" * 30)
        print(f"Fitness Score:    {getattr(saboteur, 'fitness', 'N/A')}")
        print(f"Force Segments:   {len(saboteur.forces)}")
        print(f"Max Force Peak:   {max(abs(saboteur.forces)):.2f}N")
        print("-" * 30)
        
    except Exception as e:
        print(f"Could not parse Saboteur file: {e}")

def inspect_checkpoint(checkpoint_file):
    """Extracts population-wide metrics (Species count) from a NEAT checkpoint."""
    try:
        p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        species_count = len(p.species.species)
        pop_size = len(p.population)
        
        print("-" * 30)
        print(f"CHECKPOINT ANALYSIS: {os.path.basename(checkpoint_file)}")
        print("-" * 30)
        print(f"Total Population: {pop_size}")
        print(f"Active Species:   {species_count}")
        print("-" * 30)
    except Exception as e:
        print(f"Could not parse Checkpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Metrics from Checkpoints and Brains")
    parser.add_argument("--pilot", type=str, help="Path to pilot .pkl")
    parser.add_argument("--saboteur", type=str, help="Path to saboteur .pkl")
    parser.add_argument("--checkpoint", type=str, help="Path to NEAT checkpoint")
    
    args = parser.parse_args()
    
    print("GA MISSION DATA EXTRACTION\n")
    if args.pilot: inspect_pilot(args.pilot)
    if args.saboteur: inspect_saboteur(args.saboteur)
    if args.checkpoint: inspect_checkpoint(args.checkpoint)
    
    if not (args.pilot or args.saboteur or args.checkpoint):
        parser.print_help()