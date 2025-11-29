from algorithms import Algorithms, AlgoTypes
from instance import Instance
from classifier import Classifier

class Validator:
    def validate(feature_subset:list[float], nn_type:AlgoTypes, training_data:list[Instance]) -> float:
        if len(training_data) == 0:
            return 0.0
        
        # Change to 0 indexing
        feature_indices = [i - 1 for i in feature_subset]

        correct = 0
        total = len(training_data)

        for i in range(total):
            training = []
            for j in range(total):
                # Reserve one as test data
                if i == j:
                    continue
                training.append(Validator.select_featuers(training_data[j], feature_indices))

            # Train classifer
            test = Validator.select_featuers(training_data[i], feature_indices)
            clf = Classifier(training)
            clf.train(training)
            # Test on remaining 
            predicted = clf.test(test)
            actual = training_data[i].get_class()

            if predicted == actual:
                correct += 1

        return correct / total
    
    def select_featuers(instance: Instance, feature_indices: list[int]) -> Instance:
        selected = [instance.get_feature(i) for i in feature_indices]
        return instance.with_new_features(selected)