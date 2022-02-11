import csv
import pandas as pd
import hateSpeechPrediction
import topicModelingPrediction
import sentimentAnalysisWithTextBlob


# Adding Topic, Hate Speech or Neutral Value, Sentiment Value
def createCSVToAnalyze(folder, username, field):
    df = pd.read_csv(folder + username)
    topicField = []
    hateField = []
    sentimentField =[]

    for i, row in df.iterrows():
        topicValue = topicModelingPrediction.get_topic(row[field])
        hateOrNeutralValue = hateSpeechPrediction.calculate(row[field])
        sentimentValue = sentimentAnalysisWithTextBlob.sentiment_analysis(row[field])
        topicField.append({
            'Topic': topicValue
        })
        hateField.append({
            'Value': hateOrNeutralValue
        })
        sentimentField.append({
            'Sentiment': sentimentValue
        })

    topic_score = pd.DataFrame.from_dict(topicField)
    hateOrNeutral_score = pd.DataFrame.from_dict(hateField)
    sentiment_score = pd.DataFrame.from_dict(sentimentField)
    df.join(topic_score)
    df.join(hateOrNeutral_score)
    df.join(sentiment_score)
    df.to_csv(folder + username, sep=',', index=False)


# Cleaning CSV before analyzing it
def clean(folder, username):
    df = pd.read_csv(folder + username)
    print('Start cleaning file..')

    for i, row in df.iterrows():
        row[reply] = re.sub(r"https?://[A-Za-z0-9./]*", '', row[reply])
        row[reply] = re.sub(r"@[\w]*", '', row[reply])
        row[reply] = re.sub(r"RT @[\w]*:", '', row[reply])
        row[reply] = re.sub(r"RT :", '', row[reply])
        row[reply] = re.sub('\s+', ' ', row[reply])
        row[reply] = row[reply].replace("RT", '')

        # Cleaning quotes from row Reply
        row[reply] = row[reply].replace("  ", '')
        row[reply] = row[reply].replace("   ", '')
        row[reply] = row[reply].replace("    ", '')

        # Deleting 'b' char at the beginning of the text
        if row[reply][0] == 'b':
            row[reply] = row[reply][1:]
        df.loc[i, reply] = row[reply].lower()

        # Deleting rows that have no replies
        df = df[df.Reply != "' '"]
        df = df[df.Reply != "'  '"]
        df = df[df.Reply != "'   '"]

    print('Cleaned')
    df.to_csv(folder + username, sep=',', index=False)
