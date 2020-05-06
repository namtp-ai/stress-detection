import pandas as pd
import numpy as np
import os


def root_directory():
    current_path = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_path, os.pardir))
def data_directory():
    return os.path.join(root_directory(), "hrv dataset/hrv dataset/data")

def load_train_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "train.csv")
    return pd.read_csv(in_file)
def load_test_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "test.csv")
    return pd.read_csv(in_file)

train_data  =  load_train_set()
