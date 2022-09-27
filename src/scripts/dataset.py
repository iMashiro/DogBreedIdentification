import pandas as pd

class Dataset():
    def __init__(self, dataframe):
        self.x_train = dataframe.iloc[:,0].values
        self.y_train = dataframe.iloc[:,1:].values

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        return self.x_train[index]

