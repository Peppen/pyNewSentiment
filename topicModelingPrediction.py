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


data_train = load_files(container_path='dataset\\dataset-train', load_content=True, encoding='latin1')
data_test = load_files(container_path='dataset\\dataset-test', load_content=True, encoding='latin1')


def training():
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
    with open('pickle/dataset_svc_pkl', 'wb') as files:
        pickle.dump(model, files)


# Calculate Topic result
def topic_prediction(sentence):
    all_categories_names = np.array(data_train.target_names)
    with open('pickle/dataset_svc_pkl', 'rb') as f:
        lr = pickle.load(f)
    prediction = lr.predict([sentence])
    return all_categories_names[prediction]


# Printing Topic
def get_topic(headline):
    return str(topic_prediction(headline))


if __name__ == '__main__':
    if not os.path.isfile('pickle/dataset_svc_pkl'):
        training()
    print(get_topic("It's election day"))
