import neat
import numpy as np
from Env import DrivingEnvHybrid


# --- Evaluation function for a single genome
def eval_genome(genome, config, render=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    render_mode = "human" if render else None
    env = DrivingEnvHybrid(render_mode=render_mode, num_rays=7)
    
    fitness = 0.0
    episodes = 3
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = net.activate(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            fitness += reward
            done = terminated or truncated
    
    env.close()
    return fitness / episodes


# --- Evaluation for a batch of genomes (NO rendering here for speed)
def eval_genomes(genomes, config):
    for gid, genome in genomes:
        genome.fitness = eval_genome(genome, config, render=False)


# --- Test one genome with rendering (to watch best)
def visualize_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = DrivingEnvHybrid(render_mode="human", num_rays=7)

    obs, _ = env.reset()
    done = False
    while not done:
        action = net.activate(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


def run(config_file):
    # Load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Create population
    p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Train loop manually (so we can show best genome each generation)
    num_generations = 20
    for gen in range(num_generations):
        print(f"\n****** Running generation {gen+1} ******")

        # Run one generation (NO rendering inside)
        p.run(eval_genomes, 1)

        # Get best genome of current generation
        best_genome = stats.best_genome()

        # Show graphics for BEST genome only
        print("Visualizing best genome of generation", gen+1)
        visualize_genome(best_genome, config)

    # Final winner
    winner = stats.best_genome()
    print("\nBest genome after training:", winner)
    return winner


if __name__ == "__main__":
    run("config.txt")
