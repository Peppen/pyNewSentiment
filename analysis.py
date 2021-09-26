import pandas as pd
from hatesonar.api import Sonar

results = [['Cronaca', 'Politics', 'Action', 'Drama', 'Religion']]


def hate_sonar(folder, username):
    sonar = Sonar()
    hate_rate =[]
    df = pd.read_csv(folder + username)
    for i in range(df['Reply'].shape[0]):
        score = sonar.ping(df['Reply'][i])
        hate = round(score["classes"][1]["confidence"], 2)
        if hate > 0.35:
            hate_rate.append({"Offensive": hate})
    print(len(hate_rate))


def merge(one, two, destination, username):
    df_1 = pd.read_csv(one + username)
    df_2 = pd.read_csv(two + username)
    merged = df_1.join(df_2['LDA_pred_number'])
    merged.to_csv(destination + username)


def analyze(folder, username):
    df = pd.read_csv(folder + username)
    size = 0
    negative = 0
    for i, rows in df.iterrows():
        size += 1
        if rows['Polarity'] < 0:
            negative += 1
    return str(round(negative / size * 100, 2)) + '%'


def calculate(folder, username):
    df = pd.read_csv(folder + username)
    size = 0
    hate = 0
    for i, rows in df.iterrows():
        size += 1
        if rows['Prediction'] == "['hate.speech']":
            hate += 1
    return str(round(hate / size * 100, 2)) + '%'


def compute(start, username):
    df = pd.read_csv(start + username)
    cronaca = politics = religion = drama = action = 0
    for i, row in df.iterrows():
        if row['Prediction'] == "['hate.speech']":
            if row['LDA_pred_number'] == 'Cronaca':
                cronaca += 1
            elif row['LDA_pred_number'] == 'Politics':
                politics += 1
            elif row['LDA_pred_number'] == 'Action':
                action += 1
            elif row['LDA_pred_number'] == 'Drama':
                drama += 1
            else:
                religion += 1
    results.append([cronaca, politics, religion, drama, action])
    return results


def calculate_news(start, username):
    df = pd.read_csv(start + username)
    cronaca = politics = religion = drama = action = 0
    for i, row in df.iterrows():
        if row['LDA_pred_number'] == 'Cronaca':
            cronaca += 1
        elif row['LDA_pred_number'] == 'Politics':
            politics += 1
        elif row['LDA_pred_number'] == 'Action':
            action += 1
        elif row['LDA_pred_number'] == 'Drama':
            drama += 1
        else:
            religion += 1
    results.append([cronaca, politics, religion, drama, action])
    return results

if __name__ == '__main__':
    # print(analyze('csv/analysis/', 'TIME.csv'))
    # print(calculate('csv/analysis/', 'TIME.csv'))
    # print(calculate_news('csv/analysis/', 'TIME.csv'))
    hate_sonar('replies/', 'TIME.csv')
