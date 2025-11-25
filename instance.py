class Instance:
    def __init__(self, id:int, cls: int, features:list[float]):
        self.id = id
        self.cls = cls
        self.features = features

    def get_id(self) -> int:
        return self.id
    
    def get_class(self) -> int:
        return self.cls
    
    def get_features(self) -> list[float]:
        return self.features

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Instance):
            return False
        return self.get_id() == rhs.get_id()