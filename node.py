from utils import *
from typing import List
class Node:
    def __init__(self, parent, features:List[int], score:float):
        self.parent:Node = parent
        self.features = features
        self.score = score
        self.children:List[Node] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return set(self.features) == set(other.features)

    
    def __str__(self) -> str:
        output = f"Using feature(s) {self.features_str()} accuracy is {self.score_str()}"
        return output
    
    def features_str(self) -> str:
        features_str_list = to_str_list(self.features)
        features_str = str_list_to_str(features_str_list)
        return features_str
    
    def score_str(self) -> str:
        output = f"{(self.score*100):.1f}%"
        return output
    
    def get_score(self) -> float:
        return self.score
    
    def get_children(self) -> List:
        return self.children
    
    def get_features(self) -> List:
        return self.features
    
    def empty_node():
        return Node(parent=None, features=[], score=0)
    
    def set_children(self, children:List):
        self.children = children

    def best_child(self):
        if self.children is None: # this shouldn't happen
            print("ERROR: Children have not been initialized")
            return self
        
        if len(self.children) == 0: 
            print("ERROR: There are no children of this node")
            return self

        child_bsf:Node = self.children[0]
        for child in self.children:
            if child.get_score() > child_bsf.get_score():
                child_bsf = child
        
        return child_bsf