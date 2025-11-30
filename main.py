from algorithms import Algorithms as Algos
from node import Node
from utils import *
from validator import Validator

def main():
    part_two()

def part_one():
    print(f"Welcome to cjord019/sgonz26 Feature Selection Algorithm.")

    dataset_file = input("Enter the dataset file (.txt): ")
    dataset = parse_file(dataset_file)
    
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

def part_two():
    with open("part_two_trace.txt", "w") as output:
        output.write(f"Welcome to cjord019/sgonz26 Actual Evaluation and NN-Classifier.\n")

        small = parse_file("small-test-dataset-2-2.txt")
        small_subset = [3, 5, 7]
        output.write("\nTesting small dataset with features {3,5,7}\n")
        small_acc = Validator.validate(small_subset, None, small, output = output.write)
        output.write(f"Small dataset accuracy: {small_acc:.4f} (expected: 0.89)\n")

        large = parse_file("large-test-dataset-2.txt")
        large_subset = [1, 15, 27]
        output.write("\nTesting large dataset with features {1,15,27}\n")
        large_acc = Validator.validate(large_subset, None, large, output = output.write)
        output.write(f"Large dataset accuracy: {large_acc:.4f} (expected: 0.949)\n")
    
if __name__ == "__main__":
    main()