import numpy as np
from time import time

from helpers import *

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer


# setup
training_dir = './data/train-mails'
testing_dir = './data/test-mails'

training_emails = get_email_list(training_dir)
test_emails = get_email_list(testing_dir)

vectorizer = CountVectorizer()
vocab = vectorizer.fit(training_emails)

labels_train = np.zeros(702)
labels_train[351:701] = 1
features_train = vocab.transform(training_emails).toarray()

labels_test = np.zeros(260)
labels_test[130:260] = 1
features_test = vocab.transform(test_emails).toarray()


# ml
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("\nTraining time: {}s".format(round(time()-t0, 3)))

t1 = time()
pred = clf.predict(features_test)
print("Predicting time: {}s".format(round(time()-t1, 3)))


# analysis
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print("\nAccuracy: {}".format(accuracy))

from sklearn.metrics import precision_score
precision_score = precision_score(labels_test, pred)
print("Precision: {}".format(precision_score))

from sklearn.metrics import recall_score
recall_score = recall_score(labels_test, pred)
print("Recall score: {}".format(recall_score))

from sklearn.metrics import f1_score
f1_score = f1_score(labels_test, pred)
print("F1 score: {}".format(f1_score))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(labels_test, pred)
print("\nConfusion matrix -->\n{}\n".format(confusion_matrix))
