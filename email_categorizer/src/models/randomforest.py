from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str) -> None:
        super().__init__(model_name)
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=42, class_weight='balanced_subsample')
