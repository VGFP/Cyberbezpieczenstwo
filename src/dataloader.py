# Write code that will load the data from 4 files with names: UNSW-NB15_1, UNSW-NB15_2, UNSW-NB15_3, UNSW-NB15_4 in dataset/ and with labels from NUSW-NB15_features.csv in dataset/
# Data will be loaded in pandas dataframes.

import pandas as pd
import numpy as np
import os
import itertools

class Dataloader:

    def __init__(self):
        self.data = None
        self.labels = None
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'dataset'))
        self.data_files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv'] # There is an issue with the 4th file, it is not in the correct format 'UNSW-NB15_4.csv'
        self.labels_file = 'NUSW-NB15_features_v2.csv'

    def load_data(self):
        self.data = []
        for file in self.data_files:
            print(f'Loading data from file: {file}')
            self.data.append(pd.read_csv(os.path.join(self.data_path, file), header=None))
        
        labels_arr = pd.read_csv(os.path.join(self.data_path, self.labels_file), header=None)
        labels_arr = labels_arr.values.tolist()
        self.labels = []
        for i in range(1, 50):
            self.labels.append(labels_arr[i][1])
        self.data = np.array(self.data)
        self.data = np.reshape(self.data, (700001*3, 49))
        self.df = pd.DataFrame(self.data, columns=self.labels)
        # Data cleaning
        del self.data
        del self.labels

    def get_data(self):
        return self.df


def test():


    dataloader = Dataloader()
    dataloader.load_data()
    data = dataloader.get_data()

    print(f"Data: {len(data)}")

if(__name__ == "__main__"):
    test()

