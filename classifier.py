from utils import *

class Classifier:
    def __init__(self, all_instances:list[Instance]):
        self.all_instances = all_instances # not normalized
        self.training_instances = None
        self.dimensions_minmax = None
        
    def train(self, instances:list[Instance]|list[int]):
        if len(instances) == 0:
            print("ERROR: Training set provided is empty")
            exit()

        if isinstance(instances[0], int):
            instances = self.get_instances_from_IDs(instances)
        
        normalization_results = normalize(instances)
        self.training_instances = normalization_results.get_instances()
        self.dimensions_minmax = normalization_results.get_dimensions_minmax()

    def test(self, instance:Instance|int) -> int:
        if isinstance(instance, int):
            instance_id = instance
            instance = self.get_instance_with_id(instance_id)
            if instance is None:
                print(f"ERROR: Requested instance {instance_id} is not in database")
                exit()
            
            print(f"Testing instance ID {instance.get_id()} with expected result {instance.get_class()}")
            
        if self.get_training_instances() is None or self.get_dimensions_minmax() is None:
            print(f"ERROR: Classifier has not been trained yet.")
            exit()
        
        normalized_test_input = normalize_instance(instance, self.get_dimensions_minmax())

        nearest_neighbor = None
        distance_bsf = float('inf')
        for neighbor in self.get_training_instances():
            distance = normalized_test_input.euclid_dist_to(neighbor)

            if distance < distance_bsf:
                nearest_neighbor = neighbor
                distance_bsf = distance
                print(f"{nearest_neighbor.get_class()} {distance_bsf}")
        
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
        for instance in self.get_all_instances():
            if instance.get_id() == id:
                return instance
        
        return None
        
    def get_all_instances(self) -> list[Instance]:
        return self.all_instances

    def get_training_instances(self) -> list[Instance]:
        return self.training_instances

    def get_dimensions_minmax(self) -> list[MinMax]:
        return self.dimensions_minmax