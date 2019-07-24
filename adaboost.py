import pandas as pd
import csv
import numpy as np
import nltk
from textblob import TextBlob
from matplotlib import pyplot as plt
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import operator

# print('Initial seed:', np.random.)

# Read Test and Train file
all = []
with open('./data/Headline_Testingdata.csv', encoding="utf8") as csvfile:
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
        all.append(text)
    Y = np.array(Y)[1:]

with open('./data/Headline_Trainingdata.csv', encoding="utf8") as csvfile:
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
        all.append(text)
    X = np.array(X)[1:]


# Transformation
data = [x[1] for x in X]
data_labels = [str(x[2]) for x in X]
data_test = [x[1] for x in Y]


# Lemmatize
lemma = nltk.wordnet.WordNetLemmatizer()
class StemmedVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])

# # Stem
# stemmer = nltk.stem.PorterStemmer()
# class StemmedVectorizer(TfidfVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedVectorizer, self).build_analyzer()
#         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# vectorizer = StemmedVectorizer(analyzer='word', lowercase=True)
vectorizer = StemmedVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', stop_words='english', ngram_range=(1, 1), lowercase=True)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', stop_words='english', ngram_range=(1, 1), lowercase=True)
vectorizer.fit(all)
# features = vectorizer.fit_transform(data)
features = vectorizer.transform(data)

features_nd = features.toarray() # for easy usage

# LSA transform
# svd = TruncatedSVD(n_components = 100)
# svd.fit(vectorizer.transform(all).toarray())
# # features_nd = svd.fit_transform(features_nd)
# features_nd = svd.transform(features_nd)

print(features_nd.shape)

X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.999999)

# Model selecttion
# log_model = LogisticRegression()
numEstimatorsRange = range(160, 240, 10)
learningRateRange = np.arange(0.6, 1.3, 0.1)
scoreCrossVal = list()
score_matrix = np.zeros((len(numEstimatorsRange), len(learningRateRange)))
models = list()
i = 0
for numEstimators in numEstimatorsRange:
    j = 0
    for learningRate in learningRateRange:
        models.append((numEstimators, learningRate))
        print("Running model: " + str((numEstimators, learningRate)) + "...")
        clf = AdaBoostClassifier(n_estimators=numEstimators,
                                 learning_rate=learningRate)
        scores = cross_val_score(clf, X_train, y_train)
        scoreCrossVal.append(scores.mean())
        score_matrix[i,j] = scores.mean()
        j +=1
    i += 1

index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
print("Max cross validation score: " + str(val))
optimParams = models[index]
print("Optimal number of estimators and learning rate: " + str(optimParams))

x_labels = [str(x) for x in learningRateRange]
y_labels = [str(x) for x in numEstimatorsRange]
fig, ax = plt.subplots()
heatmap = ax.pcolor(score_matrix, cmap=plt.cm.rainbow, vmin=score_matrix.min(),
                    vmax=score_matrix.max())
ax.set_xticks(np.arange(score_matrix.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(score_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(x_labels, minor=False)
ax.set_yticklabels(y_labels, minor=False)
plt.xlabel('Learning rate')
plt.ylabel('Number of estimators')
plt.title('AdaBoost cross-validation scores')
plt.colorbar(mappable=heatmap, ax=ax)
plt.show()


# Write prdiction to file
F = open("./predictions.csv", "w")
F.write('id,sentiment\n')
for i in range(len(pred)):
    F.write(str(i)+","+pred[i]+"\n")
F.close()




