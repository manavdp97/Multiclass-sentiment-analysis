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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier

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
# lemma = nltk.wordnet.WordNetLemmatizer()
# class StemmedVectorizer(TfidfVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedVectorizer, self).build_analyzer()
#         return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])

# # Stem
stemmer = nltk.stem.PorterStemmer()
class StemmedVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


# vectorizer = StemmedVectorizer(analyzer='word', lowercase=True)
# vectorizer = StemmedVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', stop_words='english', ngram_range=(1, 1), lowercase=True)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, analyzer='word', stop_words='english', lowercase=True)
vectorizer.fit(all)
# features = vectorizer.fit_transform(data)
features = vectorizer.transform(data)
features_nd = features.toarray() # for easy usage

# LSA transform
# svd = TruncatedSVD(n_components = 800)
# svd.fit(vectorizer.transform(all).toarray())
# features_nd = svd.transform(features_nd)
# xyz = 0.0
# for i in svd.explained_variance_ratio_:
#     xyz += 1.0 if i >= 0.0001 else 0.0
# print(xyz)
# features_nd = svd.fit_transform(features_nd)


print(features_nd.shape)

X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80)

# Model selecttion
# log_model = LogisticRegression()
# log_model = svm.LinearSVC()
log_model = RandomForestClassifier(bootstrap=True,
    max_depth=80,
    max_features=3,
    min_samples_leaf=5,
    min_samples_split=12,
    n_estimators=500)
# log_model =  XGBClassifier(
#  learning_rate =0.01,
#  n_estimators=500,
#  max_depth=4,
#  min_child_weight=6,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  reg_alpha=0.005,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)

# Train and assess model
    log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Transform and predict test data
data_test = vectorizer.transform(data_test).toarray()
# data_test = svd.transform(data_test)
pred = log_model.predict(data_test)


# Write prdiction to file
F = open("./predictions.csv", "w")
F.write('id,sentiment\n')
for i in range(len(pred)):
    F.write(str(i)+","+pred[i]+"\n")
F.close()




