from algorithms import Algorithms as Algos
from node import Node
from utils import *
from validator import Validator

def main():
    part_two()

def part_one():
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
    print(f"\nUsing no features and \"random\" evaluation, I get an accuracy of {(control_accuracy * 100):.1f}%\n")

    features = create_feature_list(num_features)
    print("Beginning search.\n")
    choice_result:Node = algos[algo_choice](features)

    print(f"Finished search!! The best feature subset is {choice_result.features_str()}, which has an accuracy of {choice_result.score_str()}")

def part_two():
    print(f"Welcome to cjord019/sgonz26 Actual Evaluation and NN-Classifier.")

    small = parse_file("small-test-dataset-2-2.txt")
    small_subset = [3, 5, 7]
    print("\nTesting small dataset with features {3,5,7}...")
    small_acc = Validator.validate(small_subset, None, small)
    print(f"Small dataset accuracy: {small_acc:.4f} (expected: 0.89)")

    large = parse_file("large-test-dataset-2.txt")
    large_subset = [1, 15, 27]
    print("\nTesting large dataset with features {1,15,27}...")
    large_acc = Validator.validate(large_subset, None, large)
    print(f"Large dataset accuracy: {large_acc:.4f} (expected: 0.949)")
    
if __name__ == "__main__":
    main()