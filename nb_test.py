import pandas as pd
import numpy as np
from textblob.classifiers import NaiveBayesClassifier


trainFile = 'Headline_Trainingdata.csv'
df = pd.read_csv('Headline_Trainingdata.csv', sep=',', quotechar='"')
df.columns = ["Linenum", "Phrase", "Sentiment"]
output = pd.read_csv('Headline_Testingdata.csv', sep=',', quotechar='"')
output.columns = ["Linenum", "Phrase"]
arr = df.as_matrix()

dups, indices = np.unique(arr.T[1], return_index=True)
#print(len(dups))
#print dups
dups1 = arr[indices]
dups2, counts = np.unique(dups1.T[2], return_counts = True)
for i in range(0,len(dups2)):
    print(dups2[i], counts[i])
print(len(dups2.T[1]))



cl = NaiveBayesClassifier(tb_train)
F = open("./predictions.csv", "w")
pred = []
for x in Y:
    pred.append([x[0], cl.classify(x[1])])

for i in pred:
    print(i)
    F.write(i[0]+","+i[1]+"\n")
F.close()
print(len(pred))

