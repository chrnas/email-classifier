import pandas as pd
from src.classifier_config_singleton import ClassifierConfigSingleton


class DatasetLoader():

    config: ClassifierConfigSingleton

    def __init__(self):
        self.config = ClassifierConfigSingleton()

    def read_data(self, path: str) -> pd.DataFrame:
        """This method will return the data from the given path as a pandas dataframe."""
        df = pd.read_csv(path)
        return df

    def renameColumns(self, df: pd.DataFrame):
        """Rename and preprocess columns in the DataFrame for easier access."""

        # convert the dtype object to unicode string
        df[self.config.ticket_summary] = (
            df[self.config.ticket_summary].values.astype('U')
        )
        df[self.config.interaction_content] = (
            df[self.config.interaction_content].values.astype('U')
        )

        df[self.config.type_columns] = (
            df[self.config.type_columns_names].values
        )

        df["x"] = df[self.config.interaction_content]

        df["y"] = df[self.config.classification_column]

        return df
