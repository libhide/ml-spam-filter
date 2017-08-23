import os
from collections import Counter
import numpy as np


def make_dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(train_dir, dictionary):
    files = [os.path.join(train_dir,fi) for fi in os.listdir(train_dir)]
    features_matrix = np.zeros((len(files), 3000))
    doc_id = 0;
    for fil in files:
      with open(fil) as fi:
        for i, line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              word_id = 0
              for i, d in enumerate(dictionary):
                if d[0] == word:
                  word_id = i
                  features_matrix[doc_id, word_id] = words.count(word)
        doc_id += 1
    return features_matrix