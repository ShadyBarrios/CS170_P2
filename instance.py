import math

class Instance:
    def __init__(self, id:int, cls: int, features:list[float]):
        self.id = id
        self.cls = cls
        self.features = features
    
    # returns a copy based on the list of features provided
    def with_features(self, feature_indices:list[int]):
        features = []
        for feature in feature_indices:
            if feature >= len(self.features):
                print("with_features ERROR: feature requested does not exist")
                exit()
            features.append(self.get_feature(feature))
        return Instance(self.id, self.cls, features)
    
    # returns a copy with expected normalized features
    def with_new_features(self, normalized_features:list[float]):
        return Instance(self.id, self.cls, normalized_features)

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
            running_sum += (self.features[i] - rhs.features[i])**2
        
        return math.sqrt(running_sum)