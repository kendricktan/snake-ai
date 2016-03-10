from enum import Enum

# Game parameters

# I like to play snake in a symmetrical window
GAME_WIDTH_HEIGHT = 40
NN_VISUALIZE_WIDTH_HEIGHT = 30

# Our blocksizes for the snake and apple
BLOCK_SIZE = 10
NN_VISUALIZE_SIZE = 5
NN_VISUALIZE_BLOCK_SIZE = 3

# Inputs for the Neural network
class NNObjects(Enum):
    Background = 0
    SnakeHead = 1
    SnakeBody = 2
    Apple = 3

class Directions(Enum):
    Left = -2
    Up = -1
    Down = 1
    Right = 2

# Neural Network parameters# Parameters for the network

pool = None # Our pool variable
Inputs = GAME_WIDTH_HEIGHT*GAME_WIDTH_HEIGHT # How many inputs are we supplying to the neural network
Outputs = 4 # How many outputs do we have (in our case we have 4 outputs: up, down, left, or right)
Output_Names = {0: 'Left', 1: 'Down', 2: 'Up', 3:'Right'}
Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

# Paramters for the genome
MutateConnectionsChance = 0.25
PerturbChance = 0.90
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.50
BiasMutationChance = 0.40
StepSize = 0.1
DisableMutationChance = 0.4
EnableMutationChance = 0.2

TimeoutConstant = 20

MaxNodes = 2**32
