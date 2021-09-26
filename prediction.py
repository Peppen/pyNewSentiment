import re

import csv

import json
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

if opts.all_categories:
    categories = None
else:
    categories = [
        'hate.speech',
        'neutral'
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

data = fetch_20newsgroups()
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2 + "\n")
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))

if feature_names:
    feature_names = np.asarray(feature_names)

model = make_pipeline(TfidfVectorizer(), LinearSVC())
#model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
#model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data_train.data, data_train.target)
# Predict the categories of the test data
predicted_categories = model.predict(data_test.data)


def my_predictions(my_sentence, model):
    all_categories_names = np.array(categories)
    prediction = model.predict([my_sentence])
    return all_categories_names[prediction]


def calculate(headline):
    # calculate hate speech
    hate_result = str(my_predictions(headline, model))
    print(hate_result)

    print("Accuracy:", metrics.accuracy_score(data_test.target, predicted_categories))
    print("Precision:", metrics.precision_score(data_test.target, predicted_categories, average='weighted'))
    print("Recall:", metrics.recall_score(data_test.target, predicted_categories, average='weighted'))
    print("F1-score:", metrics.f1_score(data_test.target, predicted_categories, average='weighted'))


def strip_non_ascii(string):
    """ Returns the string without non ASCII characters """
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


def prediction(start, destination, username):
    tmp = []
    results = [["Text", "Prediction"]]
    for line in open(start + '/' + username + '.json', 'r'):
        tmp.append(json.loads(line)["text"])
    for line in tmp:
        line = strip_non_ascii(line)
        line = line.lower()
        line = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', line)
        results.append([line, calculate(line)])
    with open(destination + '/' + username + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(results)


if __name__ == '__main__':
    calculate('It was very beautiful')
    #prediction('json', 'csv/prediction', 'cnn')
