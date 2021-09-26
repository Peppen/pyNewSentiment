import pandas as pd
from textblob import TextBlob
import re

tweets = []
username = 'TIME.csv'
start = 'csv/prediction/'
destination = 'csv/sentiment/'


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def sentiment_analysis(tweet):
    # Create two new columns ‘Subjectivity’ & ‘Polarity’
    tweet['Subjectivity’'] = tweet['tweet'].apply(getSubjectivity)
    tweet['Polarity'] = tweet['tweet'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        tweet['Sentiment'] = tweet['Polarity'].apply(getAnalysis)
        return tweet


def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


df = pd.read_csv(start + username)
for i, row in df.iterrows():
    row['Text'] = strip_non_ascii(row['Text'])
    row['Text'] = row['Text'].lower()
    row['Text'] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                         row['Text'])
    row['Text'] = re.sub(r'\bthats\b', 'that is', row['Text'])
    row['Text'] = re.sub(r'\bive\b', 'i have', row['Text'])
    row['Text'] = re.sub(r'\bim\b', 'i am', row['Text'])
    row['Text'] = re.sub(r'\bya\b', 'yeah', row['Text'])
    row['Text'] = re.sub(r'\bcant\b', 'can not', row['Text'])
    row['Text'] = re.sub(r'\bwont\b', 'will not', row['Text'])
    row['Text'] = re.sub(r'\bid\b', 'i would', row['Text'])
    row['Text'] = re.sub(r'wtf', 'what the fuck', row['Text'])
    row['Text'] = re.sub(r'\bwth\b', 'what the hell', row['Text'])
    row['Text'] = re.sub(r'\br\b', 'are', row['Text'])
    row['Text'] = re.sub(r'\bu\b', 'you', row['Text'])
    row['Text'] = re.sub(r'\bk\b', 'OK', row['Text'])
    row['Text'] = re.sub(r'\bsux\b', 'sucks', row['Text'])
    row['Text'] = re.sub(r'\bno+\b', 'no', row['Text'])
    row['Text'] = re.sub(r'\bcoo+\b', 'cool', row['Text'])
    tweets.append(row['Text'])

score = []
for i in range(df['Text'].shape[0]):
    polarity = getPolarity(df['Text'][i])
    subjectivity = getSubjectivity(df['Text'][i])
    score.append({'Polarity': polarity, 'Subjectivity': subjectivity})

sentiment_score = pd.DataFrame.from_dict(score)
df = df.join(sentiment_score)
df.to_csv(destination + username, index=False)
