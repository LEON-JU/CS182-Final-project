import numpy as np
import pandas as pd

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(train_data.info())
print(test_data.info())

print(train_data.describe())
print(test_data.describe())
# TODO
