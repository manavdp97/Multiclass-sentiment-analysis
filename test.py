import pandas as pd
import csv
import numpy as np
import nltk
from textblob import TextBlob
from matplotlib import pyplot as plt
from textblob.classifiers import NaiveBayesClassifier


# data = pd.read_csv('.\data\Headline_Trainingdata.csv', sep=',', header=0)
# test = pd.read_csv('.\data\Headline_Testingdata.csv', sep=',', header=0, quotechar='"', error_bad_lines=False)
# X = data.as_matrix()
# Y = test.as_matrix()
# print(Y[0, 1])

# Y = []
# with open('./Headline_Testingdata.csv', encoding="utf8") as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         if not (len(row) == 2):
#             z = []
#             z.append(row[0])
#             q = ""
#             for i in range(1,len(row)):
#                 q = q + row[i]
#             z.append(q)
#             Y.append(z)
#         else:
#             Y.append(row)
# Y = np.array(Y)[1:]
#
#
# X = []
# with open('./Headline_Trainingdata.csv', encoding="utf8") as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         if not (len(row) == 3):
#             z = []
#             z.append(row[0])
#             q = ""
#             for i in range(1,len(row) - 1):
#                 q = q + row[i]
#             z.append(q)
#             z.append(row[len(row) - 1])
#             X.append(z)
#         else:
#             X.append(row)
# X = np.array(X)[1:]
with open('./Headline_Testingdata.csv', encoding="utf8") as csvfile:
    line = csvfile.readlines()
    Y = []
    for l in line:
        z = []
        id = l[:l.find(',')]
        text = l[l.find(',') + 1:]
        text = text[text.find('"') + 1:text.rfind('"')]
        z.append(id)
        z.append(text)
        Y.append(z)
    Y = np.array(Y)[1:]

with open('./Headline_Trainingdata.csv', encoding="utf8") as csvfile:
    line = csvfile.readlines()
    X = []
    for l in line:
        z = []
        id = l[:l.find(',')]
        text = l[l.find(',') + 1:l.rfind(',')]
        sent = l[l.rfind(',') + 1:-1]
        text = text[text.find('"') + 1:text.rfind('"')]
        z.append(id)
        z.append(text)
        z.append(sent)
        X.append(z)
    X = np.array(X)[1:]

# Transformation for textblob
tb_train = [(x[1], str(x[2])) for x in X]
tb_test = [x[1] for x in Y]

# Tokenize
# train = [nltk.word_tokenize(x) for x in X[:, 1]]
# print(train[:2])

# x_val = []
# y_val = []
#
# for x in X:
#     sentiment = TextBlob(x[1])
#     # print(sentiment.sentiment.polarity)
#     x_val.append(x[2])
#     y_val.append(sentiment.sentiment.polarity)

# Plot
# plt.scatter(x_val, y_val)
# plt.show()

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

#pd.DataFrame


