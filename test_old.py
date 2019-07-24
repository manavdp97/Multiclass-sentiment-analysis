import pandas as pd
import csv
import nltk
from textblob import TextBlob
from matplotlib import pyplot as plt
from textblob.classifiers import NaiveBayesClassifier

data = pd.read_csv('.\data\\train.csv', sep=',', header=0, quotechar='"')
# test = pd.read_csv('.\data\Headline_Testingdata.csv', sep=',', header=0, quotechar='"')
X = data.as_matrix()
# Y = test.as_matrix()
# print(Y[0, 1])

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
pred = []
for x in Y:
    pred.append([x[0], cl.classify(x[1])])

# pd.DataFrame


