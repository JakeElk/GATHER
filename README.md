# GATHER: A hunter-gatherer foraging simulator
*GATHER* is an agent-based model (ABM) simulating hunter gatherer behaviour. Agents collect resources from the environemnt and return to a home with the resources. The ABM allows for manipulation of the social network topology to study how social networks result in emergent social phenomena. As is, this model allows for investigation into task perfromance and wealth inequality.
## Installation

**Note:** This document assumes you are using a UNIX-based OS (i.e. Ubuntu or MacOS)

**Note:** Please make sure that you are using a version of `ECAgent >= 0.5.0`

To use GATHER, first clone the repo and navigate to the *GATHER/* directory using your favourite terminal application.
We first need to create a virtual environment for GATHER and we do so by typing:

`> make`

into your terminal. That should create a virtual environment with all the necessary modules needed to run NeoCOOP.
Next activate the virtual environment by typing:

`> source ./venv/bin/activate`

## Running a Single Simulation:

To run an instance of GATHER, first make sure you have installed all the necessary modules and have the virtual environment activated.

To run GATHER, type the following:
```bash
> python main.py
```

For help with GATHER, type:
```bash
> python main.py -h
```

which will output:
```bash
usage: main.py [-h] [-s SIZE] [-n NEST] [-a AGENTS] [--deposit DEPOSIT] [--hdecay HDECAY] [--fdecay FDECAY] [-i ITERATIONS] [--mode MODE] [--upper UPPER]
               [-v] [-v2] [-v3] [-v4] [--seed SEED] [--move MOVE] [--gamma GAMMA] [--eta ETA] [--random RANDOM] [--detect] [--center]

options:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  The size of the environment. In all instances the environment will be a square gridworld.
  -n NEST, --nest NEST  The size of the nest. In all instances the nest will be square.
  -a AGENTS, --agents AGENTS
                        The number of agents to initialize.
  --deposit DEPOSIT     The amount of pheromone dropped by the agents.
  --hdecay HDECAY       The rate at which the home pheromone evaporates.
  --fdecay FDECAY       The rate at which the food pheromone evaporates.
  -i ITERATIONS, --iterations ITERATIONS
                        How long the simulation should run for.
  --mode MODE           What mode the environment should be initialized to
  --upper UPPER         Percenteage in dominant social group
  -v, --visualize       Whether environment images should be written to the output/ directory
  -v2, --visualize2     Whether social network topology images should be written to the output2/ directory
  -v3, --visualize3     Whether food and home pheromone images of each social group should be written to the output3/ directory
  -v4, --visualize4     Whether images of the average total, average subordinate, and average dominant group pheromones over the entire simulation should be written to the output4/ directory
  --seed SEED           Specify the seed for the Model's pseudorandom number generator
  --move MOVE           Which agent movement system to use
  --gamma GAMMA         Discount Factor for RL Systems
  --eta ETA             Learning Rate for RL Systems
  --random RANDOM       Chance for agent to take a random action.
  --detect              Should the agents detect and avoid being crowded with other agents
  --center              Should the home base be centered in the environment.

```



