from sklearn.svm import SVC
from src.modeling.base import BaseModel


class SVCModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.mdl = SVC()

    def __str__(self) -> str:
        return "SVC"
