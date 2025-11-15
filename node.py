from utils import *

class Node:
    def __init__(self, parent, features:list[int], score:float):
        self.parent:Node = parent
        self.features = features
        self.score = score
        self.children:list[Node] = None

    def __eq__(self, rhs:list[int]) -> bool:
        for feature in rhs:
            if not (feature in self.features):
                return False
        
        return True
    
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
    
    def get_children(self) -> list:
        return self.children
    
    def get_features(self) -> list:
        return self.features
    
    def empty_node():
        return Node(parent=None, features=None, score=None)
    
    def set_children(self, children:list):
        self.children = children

    def best_child(self):
        if self.children is None:
            print("ERROR: Children have not been initialized")
            return None
        
        if len(self.children) == 0:
            print("ERROR: There are no children of this node")
            return None

        child_bsf = self.children[0]
        for child in self.children:
            if child.get_score() > child_bsf.get_score():
                child_bsf = child
        
        return child_bsf