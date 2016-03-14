from enum import Enum

# Game parameters
snake = None
snakeWindow = None
GLOBAL_SPEED = 10

# I like to play snake in a symmetrical window
GAME_WIDTH_HEIGHT = 30
NN_VISUALIZE_WIDTH_HEIGHT = 20

# Our blocksizes for the snake and apple
BLOCK_SIZE = 10
NN_VISUALIZE_SIZE = 5
NN_VISUALIZE_BLOCK_SIZE = 2
PADDING = 20

# Inputs for the Neural network
class NNObjects(Enum):
    Background = -1
    SnakeBody = 1
    SnakeHead = 2
    Apple = 2

class Directions(Enum):
    Left = -2
    Up = -1
    Down = 1
    Right = 2

# Neural Network parameters# Parameters for the network

pool = None # Our pool variable
Inputs = 30 # Input 1 = Distance from us to food @ x axis, Input 2 = Distance from us to food @ x axis, Input 3-12 = 3x3 section of our left, Input 12-21 = 3x3 section of out right, Input 21-30 = 3x3 section infront of us
Outputs = 3 # How many outputs do we have (in our case we have 3 outputs: left, right, or front)
Output_Names = {0: 'Left', 1: 'Front', 2: 'Right'}
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

MaxMoveConstants = 375

MaxNodes = 2**31

class Cell:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

# Where to render our direction labels
DIR_TEXT_LOC = (250, 10)