import csv
import pandas as pd
import pickle
import io


def split(file, folder, column):
    df = pd.read_csv(file, encoding="utf8")
    for i, row in df.iterrows():
        with open(folder + str(i), 'w', encoding="utf8") as f:
            f.write(row[column])


def open_pickle(file, dest):
    with io.open(file, 'rb') as pk:
        f = pickle.load(pk)

    df = pd.DataFrame(f)
    df.to_csv(dest)



