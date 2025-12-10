from node import Node
from utils import *
from typing import List
from instance import Instance
from validator import Validator
from classifier import Classifier

class Algorithms:
    # returns random accuracy (mimicks no features and random selection)
    def no_feat(cls:Classifier) -> float:
        value = Validator.validate(cls, [], None, None)
        
        return value

    # returns Node of the feature set with highest accuracy
    def forward_selection(cls:Classifier, output=None) -> Node:
        dataset = list(cls.get_all_instances().values())

        if len(dataset) == 0:
            print("ERROR: Provided empty dataset.")
            exit()

        current_node = Node(None, [], None)   
        node_bsf = None     
        features_bsf = []
        accuracy_bsf = 0
        num_features = len(dataset[0].get_features())
        features = create_feature_list(num_features)

        while True:
            children = []

            for feature in features:
                if feature in features_bsf:
                    continue
                
                new_feature_set = [feature]
                new_feature_set.extend(features_bsf)

                score = Validator.validate(cls, new_feature_set.copy(), dataset.copy(), output)
                
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
    def backward_elimination(cls:Classifier, output=None) -> Node:
        dataset = list(cls.get_all_instances().values())

        forgiveness = 3

        if len(dataset) == 0:
            print("ERROR: Provided empty dataset.")
            exit()
        
        num_features = len(dataset[0].get_features())
        features = create_feature_list(num_features)
        current_node = Node(None, features.copy(), Validator.validate(cls, features.copy(), dataset, output))
        node_bsf = current_node
        features_bsf = features.copy()
        accuracy_bsf = current_node.get_score()

        while True:
            children = []

            # Get children by removing one feature from each 
            for feature in features_bsf:
                new_feature_set = features_bsf.copy()
                new_feature_set.remove(feature)

                score = Validator.validate(cls, new_feature_set.copy(), dataset.copy(), output)
                child = Node(current_node, new_feature_set, score)

                print(f"\t{child}")
                children.append(child)

            current_node.set_children(children)
            
            best_child = current_node.best_child()

            if best_child.get_score() >= accuracy_bsf:
                print(f"\nFeature set {best_child.features_str()} was best, accuracy is {best_child.score_str()}\n")
        
                current_node = best_child
                node_bsf = best_child
                accuracy_bsf = best_child.get_score()
                features_bsf = best_child.get_features()

            else:
                if forgiveness > 0:
                    current_node = best_child
                    print(f"\n(Warning, Accuracy has decreased!) continuing search for {forgiveness} more fails\n")
                    print(f"\nFeature set {best_child.features_str()} was best, accuracy is {best_child.score_str()}\n")
                    features_bsf = best_child.get_features()
                    forgiveness -= 1
                    
                    # Stop when one feature remains
                    if len(features_bsf) <= 1:
                        break

                    continue
                print(f"\n(Warning, Accuracy has decreased!) Ending search")
                break

            # Stop when one feature remains
            if len(features_bsf) <= 1:
                break

        return node_bsf
    
    # EC: "Original" algo will be a mix of forward selection and backward elimination search
    def hybrid_search(cls:Classifier, output=None) -> Node:
        dataset = list(cls.get_all_instances().values())

        if len(dataset) == 0:
            print("ERROR: Provided empty dataset.")
            exit()
        
        num_features = len(dataset[0].get_features())
        features = create_feature_list(num_features)

        empty_node = Node(None, [], float('-inf'))
        full_node = Node(None, features.copy(), Validator.validate(cls, features.copy(), dataset.copy(), output))

        continue_forward = True
        continue_backward = True

        current_node_forward = empty_node
        current_node_backward = full_node

        features_bsf_forward = []
        features_bsf_backward = features.copy()

        accuracy_bsf_forward = float('-inf')
        accuracy_bsf_backward = current_node_backward.get_score()

        forgiveness_backward = 3

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

                    score = Validator.validate(cls, new_feature_set.copy(), dataset.copy(), output)

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
                print(output_forward)
            #################################
            # Backwards part
            if continue_backward:
                children_backward = []

                output_backward = "BACKWARD ELIMINATION SEARCH:\n"

                for feature in features_bsf_backward:
                    new_feature_set = features_bsf_backward.copy()
                    new_feature_set.remove(feature)

                    score = Validator.validate(cls, new_feature_set.copy(), dataset.copy(), output)
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
                    if forgiveness_backward > 0:
                        current_node_backward = best_child_backward
                        output_backward += f"\n(Warning, Accuracy has decreased!) continuing search for {forgiveness_backward} more fails\n"
                        output_backward += f"\nFeature set {best_child_backward.features_str()} was best, accuracy is {best_child_backward.score_str()}\n"
                        features_bsf_backward = best_child_backward.get_features()
                        forgiveness_backward -= 1
                        
                        # Stop when one feature remains
                        if len(features_bsf_backward) <= 1:
                            continue_backward = False

                        continue
                    output_backward += f"\n(Warning, Accuracy has decreased!) Ending search"
                    continue_backward = False
                
                # Stop when one feature remains
                if len(features_bsf_backward) <= 1:
                    continue_backward = False
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