import csv

# train_file = "./data/Headline_Trainingdata.csv"
with open("./data/sample.csv") as f:
    lines = f.read().splitlines()
# train_file = "./data/sample.csv"

train = csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)

print(train)