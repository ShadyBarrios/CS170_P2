import math

class Instance:
    def __init__(self, id:int, cls: int, features:list[float]):
        self.id = id
        self.cls = cls
        self.features = features
    
    def __init__(self, original, normalized_features:list[float]):
        if not isinstance(original, Instance):
            print("ERROR: Cannot copy a non-Instance instance")
            exit()
        
        self.id = original.id
        self.cls = original.cls
        self.features = normalized_features

    def get_id(self) -> int:
        return self.id
    
    def get_class(self) -> int:
        return self.cls
    
    def get_features(self) -> list[float]:
        return self.features

    def get_feature(self, idx:int) -> float:
        if idx >= len(self.features) or idx < 0:
            print(f"ERROR: {idx} is not in range of this instance {self.id}")
            exit()

        return self.features[idx]

    def get_num_features(self) -> int:
        return len(self.features)
    
    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Instance):
            return False
        return self.get_id() == rhs.get_id()
    
    def euclid_dist_to(self, rhs) -> float:
        if not isinstance(rhs, Instance):
            return 0
    
        if len(self.features) != len(rhs.features):
            print(f"ERROR: dimensions of instance {self.id} != dimensions of instance {rhs.id}")
            return 0 
        
        num_features = len(self.features)

        running_sum = 0
        for i in range(num_features):
            running_sum = (self.features[i] - rhs.features[i])**2
        
        return math.sqrt(running_sum)