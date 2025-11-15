from node import Node
from utils import *

class Algorithms:
    # returns random accuracy (mimicks no features and random selection)
    def no_feat_random() -> float:
        return random.random()

    # returns Node of the feature set with highest accuracy
    def forward_selection(features: list[int]) -> Node:
        current_node = Node(None, [], None)
        features_bsf = []
        accuracy_bsf = 0

        while True:
            children = []

            for feature in features:
                if feature in features_bsf:
                    continue
                
                new_feature_set = [feature]
                new_feature_set.extend(features_bsf)

                score = pseudo_evaluate(new_feature_set)
                
                child = Node(current_node, new_feature_set, score)
                print(f"\t{child}")
                
                children.append(child)
            
            current_node.set_children(children)
            
            best_child = current_node.best_child()
            
            if current_node == best_child: # this happens if all features have been added
                break
            elif best_child.get_score() >= accuracy_bsf:
                print(f"\nFeature set {best_child.features_str()} was best, accuracy is {best_child.score_str()}\n")
                
                current_node = best_child
                accuracy_bsf = best_child.get_score()
                features_bsf = best_child.get_features()
            else:
                print(f"\n(Warning, Accuracy has decreased!)")
                break
        
        # current_node is the last best_child aka highest accuracy
        return current_node

    # returns Node of the feature set with the highest accuracy
    def backward_elimination(features: list[int]) -> Node:
        return Node.empty_node()