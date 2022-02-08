import pandas as pd
import csv
import re
from twarc import Twarc


t = Twarc("", "", "", "")
field = 'full_text'
reply = 'Reply'


def replies(folder, username):
    i = 0
    replies = [[reply]]
    print('Start searching data..')
    for tweet in t.search(username, max_pages=1, result_type='recent'):
        for reply in t.replies(tweet):
            replies.append([reply[field].encode("ascii", "ignore")])
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
