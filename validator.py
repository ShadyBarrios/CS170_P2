import time
from instance import Instance
from classifier import Classifier

class Validator:
    def validate(clf:Classifier, feature_subset:list[int], training_data=None, output=print) -> float:
        if training_data is None:
            training_data = list(clf.get_all_instances().values())
        
        if len(training_data) == 0:
            return 0.0
        
        # Change to 0 indexing
        feature_indices = [i - 1 for i in feature_subset]

        correct = 0
        total = len(training_data)
        train_time = 0
        test_time = 0

        if output is not None:
            output(f"Validating dataset with {total} instances\n\n")

        reduced = []

        reduce_start = time.perf_counter()
        # Select features
        for feature in training_data:
            reduced.append(Validator.select_features(feature, feature_indices))

        reduce_end = time.perf_counter()
        if output is not None:
            output(f"Selecting features took {(reduce_end - reduce_start) * 1000:.3f} ms\n \n")

        for i in range(total):
            if output is not None:
                output(f"Excluding instance {training_data[i].get_id()}\n")

            training = []
            for j in range(total):
                # Reserve an instance as test data
                if i == j:
                    continue
                training.append(reduced[j])

            # Train classifer
            test = reduced[i]
            training_start = time.perf_counter()
            # clf = Classifier(training)
            clf.train(training, feature_indices)
            training_end = time.perf_counter()
            train_time += training_end - training_start
            if output is not None:
                output(f"Classifier training took {(training_end - training_start) * 1000:.3f} ms\n")

            # Test on remaining 
            test_start = time.perf_counter()
            predicted = clf.test(test)
            test_end = time.perf_counter()
            actual = training_data[i].get_class()
            test_time += test_end - test_start
            if output is not None:
                output(f"Comparing predicted ({predicted}) and actual ({actual}) took {(test_end - test_start) * 1000:.3f} ms\n")

            if predicted == actual:
                correct += 1

            if output is not None:
                output(f"Correct so far: {correct} / {i+1}\n \n")

        if output is not None:
            output(f"Validation complete! Final accuracy: {correct / total:.4f}\n")
            output(f"Time spent training: {train_time * 1000:.3f} ms\n")
            output(f"Time spent testing: {test_time * 1000:.3f} ms\n")

        return correct / total
    
    def select_features(instance: Instance, feature_indices: list[int]) -> Instance:
        return instance.with_features(feature_indices)