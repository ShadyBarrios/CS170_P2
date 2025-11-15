from node import Node
from utils import *

class Algorithms:
    # returns random accuracy (mimicks no features and random selection)
    def no_feat_random() -> float:
        return random.random()

    # returns tuple of best features, and float of accuracy
    def forward_selection(features: list[int]) -> tuple[tuple, float]:
        return ((1,3), 0.67)

    # returns tuple of best features, and float of accuracy
    def backward_elimination(features: list[int]) -> tuple[tuple, float]:
        return ((4,1,2), 0.89)