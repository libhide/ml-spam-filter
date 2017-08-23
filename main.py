import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from helpers import *


# Create a dictionary of words with its frequency
training_dir = './data/train-mails'
dictionary = make_dictionary(training_dir)

# Prepare feature vectors per training mails and its labels
train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(training_dir, dictionary)

# Training the Naive bayes classifier
model1 = GaussianNB()
model1.fit(train_matrix, train_labels)

# Test the unseen mails for Spam
test_dir = './data/test-mails'
test_matrix = extract_features(test_dir, dictionary)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = model1.predict(test_matrix)
print(confusion_matrix(test_labels, result1))
