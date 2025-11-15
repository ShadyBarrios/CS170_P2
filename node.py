from utils import pseudo_evaluate

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
    
    def get_score(self) -> float:
        return self.score
    
    def get_children(self) -> list:
        return self.children
    
    def empty_node():
        return Node(parent=None, features=None, score=None)
    
    def create_children(self, available_features:list[int]):
        children = []
        for feature in available_features:
            features = [feature]
            features.extend(self.features)
            score = pseudo_evaluate(features)

            child = Node(self, features, score)
            children.append(child)
        self.children = children

    def best_child(self):
        if self.children is None:
            print("ERROR: best child cannot be computed prior to create_children()... returning None")
            return None
        
        if len(self.children) == 0:
            print("ERROR: There are no children of this node")
            return None

        child_bsf = self.children[0]
        for child in self.children:
            if child.get_score() > child_bsf.get_score():
                child_bsf = child
        
        return child_bsf