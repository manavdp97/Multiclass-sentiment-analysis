import numpy as np
import pandas as pd
import csv

# train_file = "./data/Headline_Trainingdata.csv"
with open("./data/sample.csv") as f:
    lines = f.read().splitlines()
# train_file = "./data/sample.csv"

train = csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)

for i in train:
    print(i[1])