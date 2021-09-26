import csv
import nltk
import json
from nltk.stem.porter import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

np.random.seed(2018)

nltk.download('wordnet')


def punctRemover(sentence):
    # list the different punctuations
    punctuations = '''!()-[]{};:'"💎\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in sentence:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


def urlRemover(sentence):
    pattern = r"http\S+"
    return re.sub(pattern, "", sentence)


def converting_LDA_genre(data):
    if data == 0:
        return "Cronaca Rosa"
    if data == 1:
        return "Politics"
    if data == 2:
        return "Tecnologia"
    if data == 3:
        return "Cronaca Nera"
    else:
        return "Religion"


def clean():
    df = pd.read_csv('tmp.csv')
    for i, row in df.iterrows():
        row["Text"] = re.sub(r"https?://[A-Za-z0-9./]*", '', row["Text"])
        row["Text"] = re.sub(r"@[\w]*", '', row["Text"])
        row["Text"] = re.sub(r"RT @[\w]*:", '', row["Text"])
        row["Text"] = re.sub(r"RT :", '', row["Text"])
        row["Text"] = row["Text"].replace("RT", '')

        # Cleaning quotes from row Reply
        row["Text"] = row["Text"].replace("  ", '')
        row["Text"] = row["Text"].replace("   ", '')
        row["Text"] = row["Text"].replace("    ", '')

        # Deleting 'b' char at the beginning of the text
        if row["Text"][0] == 'b':
            row["Text"] = row["Text"][1:]
    df.to_csv('tmp.csv', sep=',', index=False)


def calculate_topic(start, destination, username):
    tweets = []
    for line in open(start + '/' + username + '.json', 'r'):
        tweets.append(json.loads(line)["text"])
    new_tweets = [['Text']]
    for tweet in tweets:
        tweet = re.sub(r"https?://[A-Za-z0-9./]*", '', tweet)
        tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
        tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)

        new_tweets.append([urlRemover(punctRemover(tweet)).encode("ascii", "ignore")])
    with open('tmp.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(new_tweets)
    clean()
    df = pd.read_csv('tmp.csv')
    cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = cv.fit_transform(df['Text'])
    lda = LatentDirichletAllocation(n_components=5, random_state=101)
    lda_fit = lda.fit(dtm)
    for id_value, value in enumerate(lda_fit.components_):
        print(f"The topic would be {id_value}")
        print([cv.get_feature_names()[index] for index in value.argsort()[-30:]])
    results = lda_fit.transform(dtm)
    df['LDA_pred_number'] = results.argmax(axis=1)
    df['LDA_pred_number'] = df['LDA_pred_number'].apply(converting_LDA_genre)
    df.to_csv(destination + '/' + username + '.csv')


if __name__ == '__main__':
    calculate_topic('json', 'csv/topic', 'washingtonpost')
