import pandas as pd
import sqlite3

df = pd.read_csv("results (2).csv")
print("Pehle:", df.shape)

df = df.dropna()
df['date'] = pd.to_datetime(df['date'])

print("Baad mein:", df.shape)

conn = sqlite3.connect("football.db")
df.to_sql("matches", conn, if_exists="replace", index=False)
print("Done!")
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['away_score'] > row['home_score']:
        return 'Away Win'
    else:
        return 'Draw'

df['result'] = df.apply(get_result, axis=1)
print(df['result'].value_counts())
