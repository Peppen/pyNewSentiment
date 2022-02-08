import pandas as pd
from textblob import TextBlob
import re

tweets = []
text = 'Text'


# Function Getting Subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Function Getting Polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# Sentiment Analysis over Tweets
def sentiment_analysis(tweet):
    # Create two new columns ‘Subjectivity’ & ‘Polarity’
    # tweet['Subjectivity’'] = tweet['tweet'].apply(getSubjectivity)
    # tweet['Polarity'] = tweet['tweet'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        # tweet['Sentiment'] = tweet['Polarity'].apply(getAnalysis)
        return tweet


def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


# Cleaning and Analyzing Tweets
def analyze(start, destination, username):
    df = pd.read_csv(start + username)
    for i, row in df.iterrows():
        row[text] = strip_non_ascii(row[text])
        row[text] = row[text].lower()
        row[text] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                         row[text])
        row[text] = re.sub(r'\bthats\b', 'that is', row[text])
        row[text] = re.sub(r'\bive\b', 'i have', row[text])
        row[text] = re.sub(r'\bim\b', 'i am', row[text])
        row[text] = re.sub(r'\bya\b', 'yeah', row[text])
        row[text] = re.sub(r'\bcant\b', 'can not', row[text])
        row[text] = re.sub(r'\bwont\b', 'will not', row[text])
        row[text] = re.sub(r'\bid\b', 'i would', row[text])
        row[text] = re.sub(r'wtf', 'what the fuck', row[text])
        row[text] = re.sub(r'\bwth\b', 'what the hell', row[text])
        row[text] = re.sub(r'\br\b', 'are', row[text])
        row[text] = re.sub(r'\bu\b', 'you', row[text])
        row[text] = re.sub(r'\bk\b', 'OK', row[text])
        row[text] = re.sub(r'\bsux\b', 'sucks', row[text])
        row[text] = re.sub(r'\bno+\b', 'no', row[text])
        row[text] = re.sub(r'\bcoo+\b', 'cool', row[text])
        tweets.append(row[text])

    score = []
    for i in range(df[text].shape[0]):
        polarity = getPolarity(df[text][i])
        subjectivity = getSubjectivity(df[text][i])
        score.append({'Polarity': polarity, 'Subjectivity': subjectivity})

    sentiment_score = pd.DataFrame.from_dict(score)
    df = df.join(sentiment_score)
    df.to_csv(destination + username, index=False)

