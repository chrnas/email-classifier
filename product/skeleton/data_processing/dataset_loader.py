import pandas as pd

class DatasetLoader():
    def get_input_data(path): # "./data/AppGallery.csv"
        """This method will return the data from the given path as a pandas dataframe."""
        df = pd.read_csv(path)

        # convert the dtype object to unicode string
        df['Interaction content'] = df['Interaction content'].values.astype('U')
        df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

        #Optional: rename variable names for remebering easily
        df["y1"] = df["Type 1"]
        df["y2"] = df["Type 2"]
        df["y3"] = df["Type 3"]
        df["y4"] = df["Type 4"]
        df["x"] = df['Interaction content']

        df["y"] = df["y2"]
        # use all the types and not just type 2
        # df["y"] = df[["y1", "y2", "y3", "y4"]].fillna('').apply(lambda row: '|'.join(filter(None, row)), axis=1)

        return df