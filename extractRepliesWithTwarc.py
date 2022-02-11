import pandas as pd
import csv
import re
from twarc import Twarc

# Twarc credentials are entered through terminal
t = Twarc("", "", "", "")

field = 'full_text'
rep = 'Reply'


# Getting replies from each analyzed tweet
def replies(folder, username):
    i = 0
    replies = [[rep]]
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

