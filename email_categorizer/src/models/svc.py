from sklearn.svm import SVC
from .base import BaseModel


class SVCModel(BaseModel):
    def __init__(self,
                 model_name: str) -> None:
        super().__init__(model_name)
        self.mdl = SVC()
