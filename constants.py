from enum import Enum

# Game parameters
snake = None
snakeWindow = None
GLOBAL_SPEED = 1000

# I like to play snake in a symmetrical window
GAME_WIDTH_HEIGHT = 30
NN_VISUALIZE_WIDTH_HEIGHT = 20

# Our blocksizes for the snake and apple
BLOCK_SIZE = 10
NN_VISUALIZE_SIZE = 5
NN_VISUALIZE_BLOCK_SIZE = 5
PADDING = 20

# Inputs for the Neural network
class NNObjects(Enum):
    DeadEnd = -2
    Background = -1
    SnakeBody = 1
    Apple = 2

class Directions(Enum):
    Left = -2
    Up = -1
    Down = 1
    Right = 2

# Neural Network parameters# Parameters for the network

pool = None # Our pool variable
LEFT_DIMENSION_INPUTS = 1 # Inputs for our left dimension (ixi)
RIGHT_DIMENSION_INPUTS = 1 # Inputs for out right dimension (ixi)
FRONT_DIMENSION_INPUTS = 3 # Inputs for the front dimension (1xi)

# Input 0: (snake.x - apple.x)
# Input 1: (snake.y - apple.y)
# Input 2-(2+LEFT_DIMENSION_INPUTS**2): (ixi) dimension input of the snake's right area
# Inputs (2+LEFT_DIMENSION_INPUTS**2)+1 - ((2+LEFT_DIMENSION_INPUTS**2)+(RIGHT_DIMENSION_INPUTS**2)): (ixi) dimension inputs of the snake's left area
# Inputs n: (1xi) dimensional input of the snake front area
LEFT_DIMENSION_INPUTS_INDEX_END = (2+LEFT_DIMENSION_INPUTS**2)
RIGHT_DIMENSION_INPUTS_INDEX_END = (1+LEFT_DIMENSION_INPUTS_INDEX_END+(RIGHT_DIMENSION_INPUTS**2))
FRONT_DIMENSION_INPUTS_END = RIGHT_DIMENSION_INPUTS_INDEX_END+FRONT_DIMENSION_INPUTS
Inputs = 2 + (LEFT_DIMENSION_INPUTS**2) + (RIGHT_DIMENSION_INPUTS**2) + (FRONT_DIMENSION_INPUTS)

Outputs = 3 # How many outputs do we have (in our case we have 3 outputs: left, right, or front)
Output_Names = {0: 'Left', 1: 'Right', 2: 'Front'}
Population = 300
DeltaDisjoint = 2.0
DeltaWeights = 0.4
DeltaThreshold = 1.0

StaleSpecies = 15

# Paramters for the genome
MutateConnectionsChance = 0.25
PerturbChance = 0.85
CrossoverChance = 0.75
LinkMutationChance = 2.0
NodeMutationChance = 0.75
BiasMutationChance = 0.4
StepSize = 0.25
DisableMutationChance = 0.2
EnableMutationChance = 0.7

MaxMoveConstants = 100
MaxUpdateConstants = 100
MaxNodes = 2**31

class Cell:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

# Where to render our direction labels
DIR_TEXT_LOC = (250, 10)