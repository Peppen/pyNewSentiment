import numpy as np
from optparse import OptionParser
from sklearn.datasets import load_files
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from time import time
import pickle
import sys
import logging
import os
import sentimentAnalysisWithTextBlob
import extractTweetsFromTweepy

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

data_train = load_files(container_path='home\\dataset-train', load_content=True, encoding='latin1')
data_test = load_files(container_path='home\\dataset-test', load_content=True, encoding='latin1')


def size_mb(docs):
    return sum(len(s.encode('latin1')) for s in docs) / 1e6


def training():
    data_train_size_mb = size_mb(data_train.data)
    data_test_size_mb = size_mb(data_test.data)

    print("%d documents - %0.3fMB (training set)" % (
        len(data_train.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(data_test.data), data_test_size_mb))
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

    # Mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
              opts.select_chi2 + "\n")

    if feature_names:
        feature_names = np.asarray(feature_names)

    # Building Model

    # model = make_pipeline(TfidfVectorizer(), LinearSVC())
    # model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data_train.data, data_train.target)
    predicted_categories = model.predict(data_test.data)
    print("Accuracy:", metrics.accuracy_score(data_test.target, predicted_categories))
    print("Precision:", metrics.precision_score(data_test.target, predicted_categories, average='weighted'))
    print("Recall:", metrics.recall_score(data_test.target, predicted_categories, average='weighted'))
    print("F1-score:", metrics.f1_score(data_test.target, predicted_categories, average='weighted'))
    with open('pickle/model_pkl', 'wb') as files:
        pickle.dump(model, files)


# Calculate Hate Speech result
def my_predictions(my_sentence):
    all_categories_names = np.array(data_train.target_names)
    with open('pickle/model_pkl', 'rb') as f:
        lr = pickle.load(f)
    prediction = lr.predict([my_sentence])
    return all_categories_names[prediction]


# Printing result
def calculate(headline):
    # Calculate hate speech
    hate_result = str(my_predictions(headline))
    print(hate_result)


if __name__ == '__main__':
    if not os.path.isfile('pickle/model_pkl'):
        training()
