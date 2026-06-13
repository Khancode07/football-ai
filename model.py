import pandas as pd
import sqlite3
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Database se match data load karo
conn = sqlite3.connect("football.db")
df = pd.read_sql("SELECT * FROM matches", conn)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df[df['date'].dt.year >= 2008]

# Har team ka overall win rate calculate karo
win_rate = {}
for team in df['home_team'].unique():
   home = df[df['home_team'] == team]
   away = df[df['away_team'] == team]
   wins = len(home[home['home_score'] > home['away_score']]) + \
          len(away[away['away_score'] > away['home_score']])
   total = len(home) + len(away)
   win_rate[team] = wins / total if total > 0 else 0.5

# Har team ke last 10 matches ka win rate (recent form)
recent_wr = {}
for team in df['home_team'].unique():
   matches = df[(df['home_team']==team) | (df['away_team']==team)].tail(10)
   wins = 0
   for _, row in matches.iterrows():
       if row['home_team']==team and row['home_score']>row['away_score']:
           wins += 1
       elif row['away_team']==team and row['away_score']>row['home_score']:
           wins += 1
   recent_wr[team] = wins/len(matches) if len(matches)>0 else 0.5

# FIFA 22 player data se har country ka average player rating nikalo
fifa_df = pd.read_csv('players_22.csv', low_memory=False)
team_rating = fifa_df.groupby('nationality_name')['overall'].mean().round(1)
team_rating_dict = team_rating.to_dict()

# Sare features df mein add karo
df['home_winrate'] = df['home_team'].map(win_rate)
df['away_winrate'] = df['away_team'].map(win_rate)
df['winrate_diff'] = df['home_winrate'] - df['away_winrate']

df['home_recent'] = df['home_team'].map(recent_wr)
df['away_recent'] = df['away_team'].map(recent_wr)
df['recent_diff'] = df['home_recent'] - df['away_recent']

# FIFA rating feature — agar team ka data nahi toh 65 default
df['home_rating'] = df['home_team'].map(team_rating_dict).fillna(65.0)
df['away_rating'] = df['away_team'].map(team_rating_dict).fillna(65.0)
df['rating_diff'] = df['home_rating'] - df['away_rating']

# Teams aur tournament ko numbers mein convert karo (AI numbers samajhta hai)
le_home = LabelEncoder()
le_away = LabelEncoder()
le_tour = LabelEncoder()

df['home_enc'] = le_home.fit_transform(df['home_team'])
df['away_enc'] = le_away.fit_transform(df['away_team'])
df['tour_enc'] = le_tour.fit_transform(df['tournament'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Target column — 0=Home Win, 1=Away Win, 2=Draw
def get_result(row):
   if row['home_score'] > row['away_score']:
       return 0
   elif row['away_score'] > row['home_score']:
       return 1
   else:
       return 2

df['result'] = df.apply(get_result, axis=1)

# Sare features select karo — ab FIFA rating bhi hai
X = df[['home_enc', 'away_enc', 'tour_enc', 'year', 'month',
       'home_winrate', 'away_winrate', 'winrate_diff',
       'home_recent', 'away_recent', 'recent_diff',
       'home_rating', 'away_rating', 'rating_diff']]
y = df['result']

# Train/Test split — 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42)

# XGBoost model train karo
model = XGBClassifier(
   n_estimators=200,
   max_depth=6,
   learning_rate=0.05,
   subsample=0.8,
   random_state=42
)
model.fit(X_train, y_train)

# Accuracy check karo
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Model save karo pickle mein
with open('football_model.pkl', 'wb') as f:
   pickle.dump(model, f)

# Encoders aur ratings save karo
with open('encoders.pkl', 'wb') as f:
   pickle.dump({
       'le_home': le_home,
       'le_away': le_away,
       'le_tour': le_tour,
       'win_rate': win_rate,
       'recent_wr': recent_wr,
       'team_rating': team_rating_dict
   }, f)

print("Model saved!")