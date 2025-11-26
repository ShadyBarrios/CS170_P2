import random
import regex
from typing import List
from instance import Instance

# [x, y, z] -> ["x", "y", "z"]
def to_str_list(arr: List) -> List[str]:
    return [str(obj) for obj in arr]

# n -> [1, 2, ..., n]
def create_feature_list(num_features: int) -> List[int]:
    return [i+1 for i in range(num_features)]

# ["x", "y", "z"] -> "{x,y,z}"
def str_list_to_str(arr: List[str]) -> str:
    return "{" + ",".join(arr) + "}"

def pseudo_evaluate(features: List[int]) -> float:
    return random.random()

# check formatting in provided dataset using regex
def parse_file(filename:str) -> list[Instance]:
    instances:list[Instance] = []
    instance_id = 0
    size_precedence = 0

    class_format = "[1-2]\.0{7}e\+0{3}"
    feature_format = "[1-9]\.[0-9]{7}e[+-][0-9]{3}"

    if filename[-4::] != ".txt":
        print("ERROR: File must be .txt type")
        exit()

    try:
        with open(filename, "r") as file:
            while True:
                line = file.readline()
                if line == '': # EOF
                    break

                parts:list[str] = line.split()

                expected_class = parts[0]
                expected_features = " ".join(parts[1::]) # put features into one str

                checked_class = regex.findall(class_format, expected_class)
                if len(checked_class) == 0: # provided "class" does not match format
                    print(f"ERROR: Improper class format on line {instance_id + 1}")
                    exit()
                instance_class = int(float(checked_class[0])) # must convert to float first

                checked_features = regex.findall(feature_format, expected_features)
                if len(checked_features) != len(parts[1::]): # not every provided "feature" matches format
                    print(f"ERROR: Improper feature format on line {instance_id + 1}")
                    exit()
                instance_features = [float(feature) for feature in checked_features]

                # want to avoid dumb branching. so did some branchless programming here
                # will add 0 if size_precedence isn't 0
                # therefore size_precedence will only be changed during the first loop iteration
                # if size_precedence == 0:
                #    size_precedence += len(instance_features)
                # else:
                #    size_precedence += 0
                size_precedence += int(len(instance_features) * (size_precedence == 0))

                if len(instance_features) != size_precedence:
                    print("ERROR: Feature count inconsistent between instances")
                    exit()

                instance = Instance(instance_id, instance_class, instance_features)
                instances.append(instance)
                instance_id += 1
    except FileNotFoundError:
        print(f"{filename} not found. Try again.")
        exit()
    
    return instances

class MinMax:
    def __init__(self, minimum:float, maximum:float):
        self.minimum = minimum
        self.maximum = maximum

    def get_min(self) -> float:
        return self.minimum
    
    def get_max(self) -> float:
        return self.maximum
    
class NormalizationResults:
    def __init__(self, instances:list[Instance], dimensions_minmax:list[MinMax]):
        self.instances = instances
        self.dimensions_minmax = dimensions_minmax

    def get_instances(self) -> list[Instance]:
        return self.instances
    
    def get_dimensions_minmax(self) -> MinMax:
        return self.dimensions_minmax

# use min-max normalizing
def normalize(instances:list[Instance]) -> NormalizationResults:
    if len(instances) < 2: return instances
    normalized_instances = []
    dimensions = get_dimensions(instances)
    dimensions_minmax:list[MinMax] = []

    for dimension in dimensions:
        minimum = min(dimension)
        maximum = max(dimension)
        dimensions_minmax.append(MinMax(minimum, maximum)) # to normalize test inputs

    for instance in instances:
        normalized_feats = []
        for dimension in range(len(dimensions_minmax)):
            minimum = dimensions_minmax[dimension].get_min()
            maximum = dimensions_minmax[dimension].get_max()
            val = instance.get_feature(dimension)

            normalized_feat = (val - minimum) / (maximum - minimum)
            normalized_feats.append(normalized_feat)
        normalized_instances.append(instance.with_new_features(normalized_feats))


    results = NormalizationResults(normalized_instances, dimensions_minmax)
    return results

def normalize_instance(instance:Instance, dimensions_minmax:list[MinMax]) -> Instance:
    if len(instance.get_features()) != len(dimensions_minmax):
        print("ERROR: Cannot normalize instance with different amount of dimensions as dataset")
        exit()
    
    features = []
    for i in range(len(dimensions_minmax)):
        minimum = dimensions_minmax[i].get_min()
        maximum = dimensions_minmax[i].get_max()
        feat = instance.get_feature(i)
        normalized_feat = (feat - minimum) / (maximum - minimum)
        features.append(normalized_feat)
    
    normalized_instance = instance.with_new_features(features)
    return normalized_instance

# get list of features based on dimensions
def get_dimensions(instances:list[Instance]) -> list[list[float]]:
    row_size = len(instances[0].get_features())
    dimensions = []
    for col in range(row_size):
        dimension = []
        for row in range(len(instances)):
            dimension.append(instances[row].get_feature(col))
        dimensions.append(dimension)
    
    return dimensions