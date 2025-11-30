import time
from algorithms import Algorithms, AlgoTypes
from instance import Instance
from classifier import Classifier

class Validator:
    def validate(feature_subset:list[float], nn_type:AlgoTypes, training_data:list[Instance], output=None) -> float:
        if output is None:
            output = print
        
        if len(training_data) == 0:
            return 0.0
        
        # Change to 0 indexing
        feature_indices = [i - 1 for i in feature_subset]

        correct = 0
        total = len(training_data)
        train_time = 0
        test_time = 0

        output.write(f"Validating dataset with {total} instances\n\n")

        reduced = []

        reduce_start = time.perf_counter()
        # Select features
        for feature in training_data:
            reduced.append(Validator.select_featuers(feature, feature_indices))

        reduce_end = time.perf_counter()
        output.write(f"Selecting features took {(reduce_end - reduce_start) * 1000:.3f} ms\n \n")

        for i in range(total):
            output.write(f"Excluding instance {training_data[i].get_id()}\n")

            training = []
            for j in range(total):
                # Reserve an instance as test data
                if i == j:
                    continue
                training.append(reduced[j])

            # Train classifer
            test = reduced[i]
            training_start = time.perf_counter()
            clf = Classifier(training)
            clf.train(training)
            training_end = time.perf_counter()
            train_time += training_end - training_start
            output.write(f"Classifier training took {(training_end - training_start) * 1000:.3f} ms\n")

            # Test on remaining 
            test_start = time.perf_counter()
            predicted = clf.test(test)
            test_end = time.perf_counter()
            actual = training_data[i].get_class()
            test_time += test_end - test_start
            output.write(f"Comparing predicted ({predicted}) and actual ({actual}) took {(test_end - test_start) * 1000:.3f} ms\n")

            if predicted == actual:
                correct += 1

            output.write(f"Correct so far: {correct} / {i+1}\n \n")

        output.write(f"Validation complete! Final accuracy: {correct / total:.4f}\n")
        output.write(f"Time spent training: {train_time * 1000:.3f} ms\n")
        output.write(f"Time spent testing: {test_time * 1000:.3f} ms\n")

        return correct / total
    
    def select_featuers(instance: Instance, feature_indices: list[int]) -> Instance:
        selected = [instance.get_feature(i) for i in feature_indices]
        return instance.with_new_features(selected)