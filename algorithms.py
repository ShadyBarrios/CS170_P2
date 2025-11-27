from node import Node
from utils import *
from typing import List
from enum import Enum

class AlgoTypes(Enum):
    FOWARD:1
    BACKWARD:2
    BIDIRECTIONAL:3

class Algorithms:
    # returns random accuracy (mimicks no features and random selection)
    def no_feat_random() -> float:
        return random.random()

    # returns Node of the feature set with highest accuracy
    def forward_selection(features: List[int]) -> Node:
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
    def backward_elimination(features: List[int]) -> Node:
        current_node = Node(None, features.copy(), pseudo_evaluate(features))
        features_bsf = features.copy()
        accuracy_bsf = current_node.get_score()

        while True:
            children = []

            # Get children by removing one feature from each 
            for feature in features_bsf:
                new_feature_set = features_bsf.copy()
                new_feature_set.remove(feature)

                score = pseudo_evaluate(new_feature_set)
                child = Node(current_node, new_feature_set, score)

                print(f"\t{child}")
                children.append(child)

            current_node.set_children(children)
            
            best_child = current_node.best_child()

            if best_child.get_score() >= accuracy_bsf:
                print(f"\nFeature set {best_child.features_str()} was best, accuracy is {best_child.score_str()}\n")
        
                current_node = best_child
                accuracy_bsf = best_child.get_score()
                features_bsf = best_child.get_features()

            else:
                print(f"\n(Warning, Accuracy has decreased!)")
                break

            # Stop when one feature remains
            if len(features_bsf) <= 1:
                break

        return current_node
    
    # EC: "Original" algo will be a mix of forward selection and backward elimination search
    def hybrid_search(features: List[int]) -> Node:
        empty_node = Node(None, [], float('-inf'))
        full_node = Node(None, features.copy(), pseudo_evaluate(features.copy()))

        continue_forward = True
        continue_backward = True

        current_node_forward = empty_node
        current_node_backward = full_node

        features_bsf_forward = []
        features_bsf_backward = features.copy()

        accuracy_bsf_forward = float('-inf')
        accuracy_bsf_backward = current_node_backward.get_score()

        while True:
            # Forward part
            if continue_forward:
                children_forward = []

                output_forward = "FORWARD SELECTION SEARCH:\n"

                for feature in features:
                    if feature in features_bsf_forward:
                        continue

                    new_feature_set = [feature]
                    new_feature_set.extend(features_bsf_forward)

                    score = pseudo_evaluate(new_feature_set)

                    child = Node(current_node_forward, new_feature_set, score)
                    output_forward += f"\t{child}\n"

                    children_forward.append(child)

                current_node_forward.set_children(children_forward)
                best_child_forward = current_node_forward.best_child()

                if current_node_forward == best_child_forward:
                    continue_forward = False
                elif best_child_forward.get_score() >= accuracy_bsf_forward:
                    output_forward += f"\nFeature set {best_child_forward.features_str()} was best, accuracy is {best_child_forward.score_str()}\n"

                    current_node_forward = best_child_forward
                    accuracy_bsf_forward = best_child_forward.get_score()
                    features_bsf_forward = best_child_forward.get_features()
                else:
                    output_forward += f"\n(Warning, accuracy has decreased in forward selection search!)\n"
                    continue_forward = False
            #################################
            # Backwards part
            if continue_backward:
                children_backward = []

                output_backward = "BACKWARD ELIMINATION SEARCH:\n"

                for feature in features_bsf_backward:
                    new_feature_set = features_bsf_backward.copy()
                    new_feature_set.remove(feature)

                    score = pseudo_evaluate(new_feature_set)
                    child = Node(current_node_backward, new_feature_set, score)

                    output_backward += f"\t{child}\n"
                    children_backward.append(child)
                
                current_node_backward.set_children(children_backward)
                best_child_backward = current_node_backward.best_child()

                if current_node_backward == best_child_backward:
                    continue_backward = False
                elif best_child_backward.get_score() >= accuracy_bsf_backward:
                    output_backward += f"\nFeature set {best_child_backward.features_str()} was best, accuracy is {best_child_backward.score_str()}\n"

                    current_node_backward = best_child_backward
                    accuracy_bsf_backward = best_child_backward.get_score()
                    features_bsf_backward = best_child_backward.get_features()
                else:
                    output_backward += f"\n(Warning, accuracy has decreased in backward elimination search!)\n"
                    continue_backward = False

            print(output_forward)
            print(output_backward)

            # if both directions have found their best
            if not (continue_forward or continue_backward):
                break
        
        best_node = None
        if current_node_forward == current_node_backward:
            best_node = current_node_forward
        elif current_node_forward.get_score() >= current_node_backward.get_score():
            best_node = current_node_forward
        else:
            best_node = current_node_backward
        
        return best_node