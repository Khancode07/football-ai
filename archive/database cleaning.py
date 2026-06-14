import pandas as pd
import sqlite3

df = pd.read_csv("results (2).csv")
print("Pehle:", df.shape)

# Cleaning
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
print("Baad mein:", df.shape)

# Database
conn = sqlite3.connect("football.db")
df.to_sql("matches", conn, if_exists="replace", index=False)
print("Done!")

# Target column
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Home Win'
    elif row['away_score'] > row['home_score']:
        return 'Away Win'
    else:
        return 'Draw'

df['result'] = df.apply(get_result, axis=1)
print(df['result'].value_counts())

# Features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['home_team_enc'] = le.fit_transform(df['home_team'])
df['away_team_enc'] = le.fit_transform(df['away_team'])
df['tournament_enc'] = le.fit_transform(df['tournament'])

X = df[['home_team_enc', 'away_team_enc', 'tournament_enc']]
y = df['result']
print("Features ready!")
print(X.head())
# ML model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


