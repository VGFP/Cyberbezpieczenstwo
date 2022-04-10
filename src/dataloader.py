# Write code that will load the data from 4 files with names: NUSW-NB15_1, NUSW-NB15_2, NUSW-NB15_3, NUSW-NB15_4 in dataset/ and with labels from NUSW-NB15_features.csv in dataset/
# Data will be loaded in pandas dataframes.

import pandas as pd
import numpy as np
import os

class Dataloader:

    def __init__(self):
        self.data = None
        self.labels = None
        self.features = None
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  'dataset'))
        self.data_path = self.data_path.replace('Semestr 3', 'Semestr\ 3')
        print(self.data_path)
        self.data_files = ['NUSW-NB15_1.csv', 'NUSW-NB15_2.csv', 'NUSW-NB15_3.csv', 'NUSW-NB15_4.csv']
        self.labels_file = 'NUSW-NB15_features.csv'

    def load_data(self):
        self.data = []
        for file in self.data_files:
            self.data.append(pd.read_csv(os.path.join(self.data_path, file), header=None))
        self.data = np.array(self.data)
        self.labels = pd.read_csv(os.path.join(self.data_path, self.labels_file))
        self.features = self.labels.iloc[:, 1:]
        self.labels = self.labels.iloc[:, 0]

    def get_data(self):
        return self.data, self.labels, self.features

dataloader = Dataloader()
dataloader.load_data()
data, labels, features = dataloader.get_data()

print(f"Labels: {labels}")
print(f"Data: {len(data)}")


# data = load_data(os.path.join(os.path.abspath(__file__), 'dataset'))

