import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sqlite3
import numpy as np

conn = sqlite3.connect("football.db")
df = pd.read_sql("SELECT * FROM matches", conn)

le_home = LabelEncoder()
le_away = LabelEncoder()

df['home_team_enc'] = le_home.fit_transform(df['home_team'])
df['away_team_enc'] = le_away.fit_transform(df['away_team'])

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['away_score'] > row['home_score']:
        return 'Away Win'
    else:
        return 'Draw'

df['result'] = df.apply(get_result, axis=1)

X = df[['home_team_enc', 'away_team_enc']]
y = df['result']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def simulate(home, away):
    h = le_home.transform([home])[0]
    a = le_away.transform([away])[0]
    proba = model.predict_proba([[h, a]])[0]
    classes = model.classes_
    for c, p in zip(classes, proba):
        print(f"{c}: {p*100:.1f}%")

simulate("Brazil", "Argentina")