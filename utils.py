import random
from typing import List

# [x, y, z] -> ["x", "y", "z"]
def to_str_list(arr: List) -> List[str]:
    return [str(obj) for obj in arr]

# n -> [1, 2, ..., n]
def create_feature_list(num_features: int) -> List[int]:
    return [i+1 for i in range(num_features)]

# ["x", "y", "z"] -> "{x,y,z}"
def str_list_to_str(arr: List[str]) -> str:
    return "{" + ",".join(arr) + "}"

def pseudo_evaluate(features: List[int]) -> float:
    return random.random()
