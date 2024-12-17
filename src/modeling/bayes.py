from sklearn.naive_bayes import GaussianNB
from src.modeling.base import BaseModel


class Bayes(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.mdl = GaussianNB()

    def __str__(self) -> str:
        return "Bayes"
