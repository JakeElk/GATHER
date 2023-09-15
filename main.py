import argparse
import matplotlib.pyplot as plt
import sys

from src.Gather import Gather, EnvResourceComponent
from src.Agents import *

import networkx as nx
import os
import matplotlib.colors as colors
import math
import numpy.core._multiarray_umath
from matplotlib.colors import Normalize

ENV_SIZE = 30
NEST_SIZE = 4
NUM_AGENTS = 30
DEPOSIT_RATE = 0.25 
FDECAY_RATE = 0.04 
HDECAY_RATE = 0.001  
ITERATIONS = 2500
COST = 0
COST_FREQUENCY = 100000
ENV_MODE = 30
MOVE_MODE = 'QLS'
GAMMA = 0.99
ETA = 0.01
RANDOM_CHANCE = 0.0
UPPER = 0


def get_communication_network(num_agents, upper):
    
    
    print("MODE: Group-Based Hierarchy")
    net = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):    
            if i < math.ceil((upper/100) * num_agents):   
                if j < math.ceil((upper/100) * num_agents):
                    net[i][j] = 1
                else:
                    net[i][j] = 1 # was 0.5 then 1
            else:
                if j < math.ceil((upper/100) * num_agents):
                    net[i][j] = 0
                else:
                    net[i][j] = 1 # was 0.8 then 1
        net[i][i] = 1
    return net
    

def gini(data):
    sorted_x = np.sort(data)
    n = len(sorted_x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0.0 else 0.0


def parseArgs():
    """Create GATHER Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', help='The size of the environment. In all instances the environment will be a square gridworld.',
                        default=ENV_SIZE, type=int)
    parser.add_argument('-n', '--nest', help='The size of the nest. In all instances the nest will be square.',
                        default=NEST_SIZE, type=int)
    parser.add_argument('-a', '--agents', help='The number of agents to initialize.', default=NUM_AGENTS, type=int)
    parser.add_argument('--deposit', help='The amount of pheromone dropped by the agents.', default=DEPOSIT_RATE, type=float)
    parser.add_argument('--hdecay', help='The rate at which the home pheromone evaporates.', default=HDECAY_RATE, type=float)
    parser.add_argument('--fdecay', help='The rate at which the food pheromone evaporates.', default=FDECAY_RATE, type=float)
    parser.add_argument('-i', '--iterations', help='How long the simulation should run for.',
                        default=ITERATIONS, type=int)
    parser.add_argument('--mode', help='What mode the environment should be initialized to', default=ENV_MODE, type=int)
    parser.add_argument('--upper', help ='Percenteage in dominant social group', default = UPPER, type = int)
    parser.add_argument('-v', '--visualize', help='Whether environment images should be written to the output/ directory',
                        action='store_true')
    parser.add_argument('-v2', '--visualize2', help='Whether social network topology images should be written to the output2/ directory',
                        action='store_true')
    parser.add_argument('-v3', '--visualize3', help='Whether food and home pheromone images of each social group should be written to the output3/ directory',
                        action='store_true')
    parser.add_argument('-v4', '--visualize4', help='Whether images of the average total, average subordinate, and average dominant group pheromones over the entire simulation should be written to the output4/ directory',
                        action='store_true')
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)
    parser.add_argument('--move', help='Which agent movement system to use', default=MOVE_MODE, type=str)
    parser.add_argument('--gamma', help='Discount Factor for RL Systems', default=GAMMA, type=float)
    parser.add_argument('--eta', help='Learning Rate for RL Systems', default=ETA, type=float)
    parser.add_argument('--random', help='Chance for agent to take a random action.', default=RANDOM_CHANCE,
                        type=float)
    parser.add_argument('--detect', help='Should the agents detect and avoid being crowded with other agents',
                        action='store_true')
    parser.add_argument('--center', help='Should the home base be centered in the environment.',
                        action='store_true')

    return parser.parse_args()



def main():

    args = parseArgs()
    

    # specify percentage in each social class and define social network topology
    communication_network = get_communication_network(args.agents, args.upper)
    net = communication_network


    # Create Model
    model = Gather(args.size, args.nest, args.deposit, args.hdecay, args.fdecay, communication_network,
                    COST, COST_FREQUENCY, environment_mode=args.mode, seed=args.seed, move_mode=args.move,
                   gamma=args.gamma, eta=args.eta, random_chance=args.random, detect=args.detect, center=args.center)


    # Add Agents to the environment
    for i in range(args.agents):
        model.environment.add_agent(AntAgent(i, model), *model.random.choice(model.home_locs))


    # print paramters 
    print("Agents: " + str(args.agents))
    print("Env size: " + str(args.size))
    print("Dominant Group Percentage: " + str(args.upper))
    print("Subordinate Group Percentage: " + str(100 - args.upper))
    

    # Run Model
    agent_val = len(model.resource_distribution) + 1
    for i in range(args.iterations):

        wealth_upper = 0
        wealth_lower = 0

        model.execute()

        for agent in model.environment:
            if agent.id < math.ceil(args.agents*(args.upper/100)):
                wealth_upper+=agent[ResourceComponent].wealth
            else:
                wealth_lower+=agent[ResourceComponent].wealth


        wealth_arr = np.array([agent[ResourceComponent].wealth for agent in model.environment])
        print(f'\r{i+1}/{ITERATIONS} - Total Collected: {model.environment[EnvResourceComponent].resources} Dominant Group Wealth: {wealth_upper} Subordinate Group Wealth: {wealth_lower} Gini: {round(gini(wealth_arr),4)}', \
              file=sys.stdout, end = '\r', flush=True)
        
        if args.visualize:
            # Will generate a series of figures of the environment.
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
            img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(args.size, args.size)

            for agent in model.environment:
                x, y = agent[ENV.PositionComponent].xy()
                img[y][x] = agent_val

                # display agents of different groups in different colours
                if agent.id < math.ceil(args.agents*(args.upper/100)):
                    img[y][x] = agent_val + 1  
                else:
                    img[y][x] = agent_val + 2 


            ax1.imshow(img, cmap='Set1')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Environment')
            fcells = np.zeros(model.environment.width ** 2)
            hcells = np.zeros(model.environment.width ** 2)

            for agent in model.environment:
                fcells += agent[PheromoneComponent].f_pheromones
                hcells += agent[PheromoneComponent].h_pheromones

            img = fcells.reshape(args.size, args.size)
            ax2.imshow(img)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Food Pheromones')

            img = hcells.reshape(args.size, args.size)
            ax3.imshow(img)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_title('Home Pheromones')

            fig.suptitle(f'Iteration {i}')
            fig.savefig(f'./output/iteration_{i}.png')
            plt.close(fig)

        if args.visualize2:
            # display social network topology with wealth per each agent
            G = nx.Graph()
            G.add_nodes_from(range(args.agents))

            for x in range(args.agents):
                for z in range(x + 1, args.agents):
                    if net[x][z] > 0:
                        G.add_edge(x, z, weight=net[x][z])  # Corrected indexing

            # Retrieve the weights from the edges
            edge_weights = [G.edges[x, z]['weight'] for x, z in G.edges()]
            normalized_weights = [weight / max(edge_weights) for weight in edge_weights]
            cmap = plt.cm.viridis

            # specify number of red and green nodes (red are dominant agnets, green are subordinate)
            numRed = math.ceil(args.agents*(args.upper/100))
            numGreen = (args.agents - numRed)
            node_colors = ['red'] * (numRed) + ['green'] * (numGreen)
            node_color = [node_colors[node_id] for node_id in G.nodes()] # Use the node_colors list to assign colors based on node IDs

            wealths = [0]*args.agents
            for agent in model.environment:
                wealths[agent.id] = agent[ResourceComponent].wealth

            labels = {node_id: wealths[node_id] for node_id in G.nodes()}
                

            fig, ax = plt.subplots()
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=800, ax=ax)
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')


            # Create a list of edges
            edgelist = [(u, v) for u, v in G.edges()]


            # Draw edges (can handle different edge weights)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edgelist,
                edge_color=normalized_weights,
                edge_cmap=cmap,
                width=2,
                alpha=0.7,
                ax=ax
            )

            fig.suptitle(f'Iteration {i}')
            fig.savefig(f'./output2/iteration_{i}.png')
            plt.close(fig)

        if args.visualize3:

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

            fcellsDom = np.zeros(model.environment.width ** 2)
            hcellsDom = np.zeros(model.environment.width ** 2)
            fcellsSub = np.zeros(model.environment.width ** 2)
            hcellsSub = np.zeros(model.environment.width ** 2)

            for agent in model.environment:

                if agent.id < math.ceil(args.agents*(args.upper/100)): 
                    fcellsDom += agent[PheromoneComponent].f_pheromones
                    hcellsDom += agent[PheromoneComponent].h_pheromones

                else:
                    fcellsSub += agent[PheromoneComponent].f_pheromones
                    hcellsSub += agent[PheromoneComponent].h_pheromones

            img = fcellsDom.reshape(args.size, args.size)
            ax1.imshow(img)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')

            img = hcellsDom.reshape(args.size, args.size)
            ax2.imshow(img)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')

            img = fcellsSub.reshape(args.size, args.size)
            ax3.imshow(img)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')

            img = hcellsSub.reshape(args.size, args.size)
            ax4.imshow(img)
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')

            ax1.set_title('Dominant Food Pheromones')
            ax2.set_title('Dominant Home Pheremones')
            ax3.set_title('Subordinate Food Pheremones')
            ax4.set_title('Subordinate Home Pheremones')

            fig.suptitle(f'Iteration {i}')
            fig.savefig(f'./output3/iteration_{i}.png')
            plt.close(fig)


        if args.visualize4:
            if i == 0:
                hcells2 = np.zeros(model.environment.width ** 2)
                hcells3 = np.zeros(model.environment.width ** 2)
                hcells4 = np.zeros(model.environment.width ** 2)
                
            elif i == (args.iterations - 1):

                fig, (ax1,ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
                norm = Normalize(vmin=0, vmax=1)
                max_value = np.amax(hcells2)
                
                # First subplot
                hcells2 = (hcells2 / max_value)
                img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(args.size, args.size)
                img = hcells2.reshape(args.size, args.size)
                ax1.imshow(img, interpolation='nearest', norm=norm)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_title('Total Home Pheromones')

                # Second subplot
                hcells3 = (hcells3 / max_value)
                img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(args.size, args.size)
                img = hcells3.reshape(args.size, args.size)
                ax2.imshow(img, interpolation='nearest', norm=norm)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_title('Dominant Home Pheromones')

                # Third subplot:
                hcells4 = (hcells4 / max_value)
                img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(args.size, args.size)
                img = hcells4.reshape(args.size, args.size)
                ax3.imshow(img, interpolation='nearest', norm=norm)
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_title('Subordinate Home Pheromones') 

                fig.savefig(f'./output4/iteration_{i}.png')
                plt.close(fig)

            else:
                for agent in model.environment:
                    hcells2 += agent[PheromoneComponent].h_pheromones
                    if agent.id < math.ceil(args.agents*(args.upper/100)):
                        hcells3 += agent[PheromoneComponent].h_pheromones
                    else:
                        hcells4 += agent[PheromoneComponent].h_pheromones

    print()


if __name__ == '__main__':
    main()
