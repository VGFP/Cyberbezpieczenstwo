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
        self.features = None
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'dataset'))
        self.data_files = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv'] # There is an issue with the 4th file, it is not in the correct format 'UNSW-NB15_4.csv'
        self.labels_file = 'NUSW-NB15_features_v2.csv'

    def load_data(self):
        self.data = []
        for file in self.data_files:
            print(f'Loading data from file: {file}')
            self.data.append(pd.read_csv(os.path.join(self.data_path, file), header=None))
        with open(os.path.join(self.data_path, self.labels_file), 'r') as f:
            self.labels = f.readlines()
        self.data = np.array(self.data)
        self.data = np.reshape(self.data, (700001*3, 49))
        self.df = pd.DataFrame(self.data)

    def get_data(self):
        return self.data, self.labels, self.features


def test():


    dataloader = Dataloader()
    dataloader.load_data()
    data, labels, features = dataloader.get_data()

    # print(f"Labels: {labels}")
    # print(f"Data: {len(data)}")

if(__name__ == "__main__"):
    test()

