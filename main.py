from algorithms import Algorithms as Algos
from node import Node
from utils import *

#TODO: remake this to test validator class
def main():
    print(f"Welcome to cjord019/sgonz26 Feature Selection Algorithm.")
    try:
        num_features = int(input("Please enter total number of features: "))
        if num_features < 1:
            print("Invalid number of features, defaulting to four features...")
            num_features = 4
    except ValueError:
        print("Improper input type, defaulting to four features...")
        num_features = 4
    
    print("Type the number of the algorithm you want to run.")
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination")
    print("\t3) Hybrid Search (Forward Selection + Backward Elimination)")
    try:
        algo_choice = int(input("Choice (1-3): "))
        if algo_choice < 1 or algo_choice > 3:
            print("Improper algo selection, defaulting to Forward Selection...")
            algo_choice = 1
    except ValueError:
        print("Improper input type, defaulting to Forward Selection...")
        algo_choice = 1

    algos:tuple[function] = (Algos.no_feat_random, Algos.forward_selection, Algos.backward_elimination, Algos.hybrid_search)

    control_accuracy:float = algos[0]()
    print(f"\nUsing no features and \"random\" evaluation, I get an accuracy of {(control_accuracy * 100):.1f}%\n")

    features = create_feature_list(num_features)
    print("Beginning search.\n")
    choice_result:Node = algos[algo_choice](features)

    print(f"Finished search!! The best feature subset is {choice_result.features_str()}, which has an accuracy of {choice_result.score_str()}")
    
if __name__ == "__main__":
    main()