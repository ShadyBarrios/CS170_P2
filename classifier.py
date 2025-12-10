from utils import *
from enum import Enum

class Normalize(Enum):
    Z_SCORE = 1
    NONE = 2

class Classifier:
    def __init__(self, all_instances:list[Instance], normalization:Normalize):
        self.all_instances = {}
        self.training_instances = {}


        if normalization == Normalize.Z_SCORE:
            self.all_instances = normalize(all_instances)
        else:
            self.all_instances = to_dict(all_instances)
        
    def train(self, instances:list[Instance]|list[int], features:list[int]):
        if len(instances) == 0:
            print("ERROR: Training set provided is empty")
            exit()

        ids = instances
        if isinstance(instances[0], Instance):
            ids = [instance.get_id() for instance in instances]
        
        self.training_instances = {}
        for id in ids:
            instance = self.get_instance_with_id(id)
            if instance is None:
                print("Train error: provided ID does not exist in dataset")
            self.training_instances[id] = instance.with_features(features)

    def test(self, instance:Instance|int) -> int:
        if isinstance(instance, int):
            instance_id = instance
            instance = self.get_instance_with_id(instance_id)
            if instance is None:
                print(f"ERROR: Requested instance {instance_id} is not in database")
                exit()
            
            # print(f"Testing instance ID {instance.get_id()} with expected result {instance.get_class()}")
            
        if self.get_training_instances() is None:
            print(f"ERROR: Classifier has not been trained yet.")
            exit()

        nearest_neighbor = None
        distance_bsf = float('inf')

        for neighbor in list(self.get_training_instances().values()):
            distance = instance.euclid_dist_to(neighbor)

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
            return None
        return instance
        
    def get_all_instances(self) -> dict[int, Instance]:
        return self.all_instances

    def get_training_instances(self) -> dict[int, Instance]:
        return self.training_instances