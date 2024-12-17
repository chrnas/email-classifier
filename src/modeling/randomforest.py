from sklearn.ensemble import RandomForestClassifier
from src.modeling.base import BaseModel


class RandomForest(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.mdl = RandomForestClassifier(
            n_estimators=1000, random_state=42, class_weight='balanced_subsample')

    def __str__(self) -> str:
        return "RandomForest"
