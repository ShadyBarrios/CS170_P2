from algorithms import Algorithms as Algos
from utils import *

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
    print("\t3) Our Special Algorithm (NOT AVAILABLE - WORK-IN-PROGRESS)")
    try:
        algo_choice = int(input("Choice (1-2): "))
        if algo_choice != 1 and algo_choice != 2:
            print("Improper algo selection, defaulting to Forward Selection...")
            algo_choice = 1
    except ValueError:
        print("Improper input type, defaulting to Forward Selection...")
        algo_choice = 1

    algos:tuple[function] = (Algos.no_feat_random, Algos.forward_selection, Algos.backward_elimination)

    control_accuracy:float = algos[0]()
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {(control_accuracy * 100):.1f}%")

    features = create_feature_list(num_features)
    choice_result:tuple[tuple, float] = algos[algo_choice](features)

    best_features:tuple[int] = choice_result[0]
    best_accuracy:float = choice_result[1]
    
    best_features_str:list[str] = to_str_list(best_features)
    best_features_output = str_list_to_str(best_features_str)

    print(f"Finished search!! The best feature subset is {best_features_output}, which has an accuracy of {(best_accuracy*100):.1f}%")
    
if __name__ == "__main__":
    main()