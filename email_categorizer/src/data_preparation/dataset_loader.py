import pandas as pd
from classifier_config_singleton import ClassifierConfigSingleton


class DatasetLoader():

    config_manager: ClassifierConfigSingleton

    def __init__(self):
        self.config_manager = ClassifierConfigSingleton()

    def read_data(self, path: str) -> pd.DataFrame:
        """This method will return the data from the given path as a pandas dataframe."""
        df = pd.read_csv(path)
        return df

    def renameColumns(self, df: pd.DataFrame):
        """Rename and preprocess columns in the DataFrame for easier access."""

        # convert the dtype object to unicode string
        df[self.config_manager.input_columns["ticket_summary"]] = (
            df['Interaction content'].values.astype('U')
        )
        df[self.config_manager.input_columns["interaction_content"]] = (
            df['Ticket Summary'].values.astype('U')
        )

        #for i in range(len(self.config_manager.type_columns)):
        #    df[self.config_manager.type_columns[i]] = (
        #        df[self.config_manager.type_columns_names[i]]
        #    )

        df[self.config_manager.type_columns] = (
            df[self.config_manager.type_columns_names].values
        )

        df["x"] = df[self.config_manager.input_columns["interaction_content"]]

        df["y"] = df[self.config_manager.classification_column]

        return df
