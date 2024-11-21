from sklearn.naive_bayes import GaussianNB
from .base import BaseModel

class Bayes(BaseModel):
    def __init__(self,
                 model_name: str) -> None:
        super().__init__(model_name)
        self.mdl = GaussianNB()
