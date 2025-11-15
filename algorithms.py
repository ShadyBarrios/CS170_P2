from node import Node

class Algorithms:
    # returns accuracy
    def no_feat_random() -> float:
        return 0.45

    # returns tuple of best features, and float of accuracy
    def forward_selection(features: list[int]) -> tuple[tuple, float]:
        return ((1,3), 0.67)

    # returns tuple of best features, and float of accuracy
    def backward_elimination(features: list[int]) -> tuple[tuple, float]:
        return ((4,1,2), 0.89)