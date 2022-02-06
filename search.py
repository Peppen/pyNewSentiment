import pandas as pd

import csv
import re
from twarc import Twarc

t = Twarc("", "", "", "")


def replies(folder, username):
    i = 0
    replies = [['Reply']]
    print('Start searching data..')
    for tweet in t.search(username, max_pages=1, result_type='recent'):
        for reply in t.replies(tweet):
            replies.append([reply["full_text"].encode("ascii", "ignore")])
            if i == 100:
                break
            i += 1

    outfile = username + ".csv"
    print("Writing to " + folder + '/' + outfile)
    with open(folder + '/' + outfile, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(replies)


def clean(folder, username):
    df = pd.read_csv(folder + username)
    print('Start cleaning file..')
    for i, row in df.iterrows():
        row["Reply"] = re.sub(r"https?://[A-Za-z0-9./]*", '', row["Reply"])
        row["Reply"] = re.sub(r"@[\w]*", '', row["Reply"])
        row["Reply"] = re.sub(r"RT @[\w]*:", '', row["Reply"])
        row["Reply"] = re.sub(r"RT :", '', row["Reply"])
        row["Reply"] = re.sub('\s+', ' ', row["Reply"])
        row["Reply"] = row["Reply"].replace("RT", '')

        # Cleaning quotes from row Reply
        row["Reply"] = row["Reply"].replace("  ", '')
        row["Reply"] = row["Reply"].replace("   ", '')
        row["Reply"] = row["Reply"].replace("    ", '')

        # Deleting 'b' char at the beginning of the text
        if row["Reply"][0] == 'b':
            row["Reply"] = row["Reply"][1:]
        df.loc[i, "Reply"] = row["Reply"].lower()

        # Deleting rows that have no replies
        df = df[df.Reply != "' '"]
        df = df[df.Reply != "'  '"]
        df = df[df.Reply != "'   '"]
    print('Cleaned')
    df.to_csv(folder + username, sep=',', index=False)
