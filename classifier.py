from utils import *

class Classifier:
    def __init__(self, all_instances:list[Instance]):
        self.all_instances = all_instances # not normalized
        
    def Train(self, instances:list[Instance]|list[int]):
        if len(instances) == 0:
            print("ERROR: Training set provided is empty")
            exit()

        if isinstance(instances[0], int):
            instances = self.get_instances_from_IDs(instances)
        
        normalization_results = normalize(instances)
        self.training_instances = normalization_results.get_instances()
        self.dimensions_minmax = normalization_results.get_dimStats()

    def Test(self, instance:Instance|int) -> int:
        if isinstance(instance, int):
            instance_id = instance
            instance = self.find_instance_with_id(instance_id)
            if instance is None:
                print(f"ERROR: Requested instance {instance_id} is not in database")
                exit()
        
        normalized_instance = normalize_instance(instance, self.get_dimensions_minmax())
        
        pass
    
    def get_instances_from_IDs(self, IDs:list[int]) -> list[Instance]:
        instances = []
        for id in IDs:
            instance = self.find_instance_with_id(id)
            if instance is None:
                print("ERROR: Requested ID is not in dataset")
                exit()
            instances.append(instance)
        return instances

    def find_instance_with_id(self, id:int) -> Instance|None:
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