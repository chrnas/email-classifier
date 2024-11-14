from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self) -> None:
        ...

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.get_y_train())
        print("bayes")

    def predict(self, data) -> None:
        predictions = self.mdl.predict(data.get_X_test())
        self.predictions = predictions

    @abstractmethod
    def data_transform(self) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
