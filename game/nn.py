import math
from .constants import *

def sigmoid(x):
    return 1/(1+math.exp(-x))
