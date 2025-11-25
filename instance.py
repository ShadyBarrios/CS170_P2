class Instance:
    def __init__(self, cls: int, features:list[float]):
        self.cls = cls
        self.features = features

    def get_class(self) -> int:
        return self.cls
    
    def get_features(self) -> list[float]:
        return self.features