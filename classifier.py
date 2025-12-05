from utils import *

class Classifier:
    # all instances is not normalized
    def __init__(self, all_instances:list[Instance]):
        if len(all_instances) == 0:
            print("ERROR: Set provided is empty")
            exit()

        normalization_results = normalize(all_instances)
        self.all_instances = normalization_results.get_instances()
        self.dimensions_stats = normalization_results.get_dimensions_stats()

    def train(self, instances:list[int]):
        if len(instances) == 0:
            print("ERROR: Training set provided is empty")
            exit()

        if isinstance(instances[0], int):
            self.training_instances = self.get_instances_from_IDs(instances)
        else:
            print("ERROR: Invalid training supply, must be list of ID's")
            exit()

    def test(self, instance:Instance|int) -> int:
        if isinstance(instance, int):
            instance_id = instance
            instance = self.get_instance_with_id(instance_id)
            if instance is None:
                print(f"ERROR: Requested instance {instance_id} is not in database")
                exit()
            
            # print(f"Testing instance ID {instance.get_id()} with expected result {instance.get_class()}")
            
        if self.get_training_instances() is None or self.get_dimensions_stats() is None:
            print(f"ERROR: Classifier has not been trained yet.")
            exit()
        
        normalized_test_input = normalize_instance(instance, self.get_dimensions_stats())
        # test_input = instance

        nearest_neighbor = None
        distance_bsf = float('inf')
        for neighbor in self.get_training_instances():
            distance = normalized_test_input.euclid_dist_to(neighbor)
            # distance = test_input.euclid_dist_to(neighbor)

            if distance < distance_bsf:
                nearest_neighbor = neighbor
                distance_bsf = distance
                # print(f"{nearest_neighbor.get_class()} {distance_bsf}")
        
        predicted_class = nearest_neighbor.get_class()
        # print(f"Normalized input features {normalized_test_input.get_features()}")
        # print(f"Nearest neighbor {nearest_neighbor.get_id()}, distance {distance_bsf}, features {nearest_neighbor.get_features()}")
        return predicted_class
    
    def get_instances_from_IDs(self, IDs:list[int]) -> list[Instance]:
        instances = []
        for id in IDs:
            instance = self.get_instance_with_id(id)
            if instance is None:
                print("ERROR: Requested ID is not in dataset")
                exit()
            instances.append(instance)
        return instances

    def get_instance_with_id(self, id:int) -> Instance|None:
        instance = self.all_instances[id]

        if instance is None:
            print(f"Instance {id} does not exist in classifier dataset")
            exit()
        
        return instance
        
    def get_all_instances(self) -> list[Instance]:
        return self.all_instances

    def get_training_instances(self) -> list[Instance]:
        return self.training_instances

    def get_dimensions_stats(self) -> list[DimensionStats]:
        return self.dimensions_stats