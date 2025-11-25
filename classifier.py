from instance import Instance
from utils import normalize

class Classifier:
    def __init__(self, instances:list[Instance]):
        self.instances = instances
        
    def Train(self, instances:list[Instance]):
        normalized_instances = normalize(instances)
        self.__init__(normalized_instances)

    def Test(self, instance:Instance|int) -> int:
        pass