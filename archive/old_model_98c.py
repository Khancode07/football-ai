import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  WORLD CUP 2026 PREDICTOR MODEL TRAINING")
print("="*60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
try:
    df_results = pd.read_csv('archive/results (2).csv')
    df_shootouts = pd.read_csv('archive/shootouts (2).csv')
except FileNotFoundError:
    print("❌ Error: Datasets not found in 'archive/' directory.")
    print("Please ensure results (2).csv and shootouts (2).csv exist.")
    exit(1)

# Ensure datetime format and sort chronologically
df_results['date'] = pd.to_datetime(df_results['date'])
df_results = df_results[df_results['date'].dt.year >= 2000].sort_values('date').reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 2. TOURNAMENT TIER WEIGHTS
# ─────────────────────────────────────────────────────────────
TOURNAMENT_TIER = {
    'FIFA World Cup': 3.0,
    'Copa América': 2.5,
    'UEFA Euro': 2.5,
    'Africa Cup of Nations': 2.0,
    'AFC Asian Cup': 2.0,
    'CONCACAF Gold Cup': 2.0,
    'UEFA Nations League': 1.8,
    'FIFA World Cup qualification': 1.5,
    'Friendly': 0.7,
}

def get_tour_weight(tournament):
    for key, w in TOURNAMENT_TIER.items():
        if key.lower() in tournament.lower():
            return w
    return 1.0

df_results['tour_tier'] = df_results['tournament'].apply(get_tour_weight)

# ─────────────────────────────────────────────────────────────
# 3. COMPUTE TARGET (HOME WIN, DRAW, AWAY WIN)
# ─────────────────────────────────────────────────────────────
def get_result(row):
    if row['home_score'] > row['away_score']: return 'Home Win'
    if row['home_score'] < row['away_score']: return 'Away Win'
    return 'Draw'

df_results['result'] = df_results.apply(get_result, axis=1)

# ─────────────────────────────────────────────────────────────
# 4. ELO RATINGS ENGINE
# ─────────────────────────────────────────────────────────────
# Elo must be calculated on ALL matches, including friendlies, to be accurate.
elo_dict = {}
ELO_START = 1500
K_FACTOR = 40

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

h_elo_list = []
a_elo_list = []

print("⚡ Computing Elo ratings...")
for idx, row in df_results.iterrows():
    home = row['home_team']
    away = row['away_team']
    
    h_elo = elo_dict.get(home, ELO_START)
    a_elo = elo_dict.get(away, ELO_START)
    
    h_elo_list.append(h_elo)
    a_elo_list.append(a_elo)
    
    if row['result'] == 'Home Win':
        score_h, score_a = 1, 0
    elif row['result'] == 'Away Win':
        score_h, score_a = 0, 1
    else:
        score_h, score_a = 0.5, 0.5
        
    exp_h = expected_score(h_elo, a_elo)
    exp_a = expected_score(a_elo, h_elo)
    
    # Update Elo, scaled by tournament importance and goal difference
    gd = abs(row['home_score'] - row['away_score'])
    margin_multiplier = np.log(gd + 1) if gd > 0 else 1.0
    k = K_FACTOR * row['tour_tier'] * margin_multiplier
    
    elo_dict[home] = h_elo + k * (score_h - exp_h)
    elo_dict[away] = a_elo + k * (score_a - exp_a)

df_results['home_elo'] = h_elo_list
df_results['away_elo'] = a_elo_list

# ─────────────────────────────────────────────────────────────
# 5. ROLLING STATS (FORM & H2H)
# ─────────────────────────────────────────────────────────────
print("⚡ Computing rolling stats & form...")
team_stats = {}
h2h_history = {}

# We'll store rolling features for the dataframe here
feats = {'h_form':[], 'a_form':[], 'h_gd':[], 'a_gd':[], 'h2h_hw':[], 'h2h_aw':[], 'h2h_d':[]}

for idx, row in df_results.iterrows():
    home = row['home_team']
    away = row['away_team']
    
    hs = team_stats.setdefault(home, {'history': [], 'gd': []})
    as_ = team_stats.setdefault(away, {'history': [], 'gd': []})
    
    # Form: win rate in last 10 games
    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    a_form = sum([1 for r in as_['history'][-10:] if r == 'W']) / 10.0 if as_['history'] else 0.45
    
    # Goal Difference Average
    h_gd = np.mean(hs['gd'][-10:]) if hs['gd'] else 0.0
    a_gd = np.mean(as_['gd'][-10:]) if as_['gd'] else 0.0
    
    # H2H (Last 5 meetings)
    pair = tuple(sorted([home, away]))
    hist = h2h_history.setdefault(pair, [])
    recent_h2h = hist[-5:]
    h2h_hw = sum([1 for m in recent_h2h if m == home])
    h2h_aw = sum([1 for m in recent_h2h if m == away])
    h2h_d  = sum([1 for m in recent_h2h if m == 'Draw'])
    
    feats['h_form'].append(h_form)
    feats['a_form'].append(a_form)
    feats['h_gd'].append(h_gd)
    feats['a_gd'].append(a_gd)
    feats['h2h_hw'].append(h2h_hw if home == pair[0] else h2h_aw)
    feats['h2h_aw'].append(h2h_aw if away == pair[0] else h2h_hw)
    feats['h2h_d'].append(h2h_d)
    
    # Update state for NEXT match
    if row['result'] == 'Home Win':
        hs['history'].append('W'); as_['history'].append('L'); hist.append(home)
    elif row['result'] == 'Away Win':
        hs['history'].append('L'); as_['history'].append('W'); hist.append(away)
    else:
        hs['history'].append('D'); as_['history'].append('D'); hist.append('Draw')
        
    gd = row['home_score'] - row['away_score']
    hs['gd'].append(gd)
    as_['gd'].append(-gd)

df_results['h_form'] = feats['h_form']
df_results['a_form'] = feats['a_form']
df_results['h_gd'] = feats['h_gd']
df_results['a_gd'] = feats['a_gd']
df_results['h2h_hw'] = feats['h2h_hw']
df_results['h2h_aw'] = feats['h2h_aw']
df_results['h2h_d'] = feats['h2h_d']

# ─────────────────────────────────────────────────────────────
# 6. FILTER COMPETITIVE MATCHES & ENCODE
# ─────────────────────────────────────────────────────────────
# Drop friendlies to remove noise from ML training
df_ml = df_results[df_results['tournament'] != 'Friendly'].copy()

print(f"🎯 Training on competitive matches only: {len(df_ml)}")

le_home = LabelEncoder()
le_away = LabelEncoder()
le_tour = LabelEncoder()

df_ml['home_enc'] = le_home.fit_transform(df_ml['home_team'])
df_ml['away_enc'] = le_away.fit_transform(df_ml['away_team'])
df_ml['tour_enc'] = le_tour.fit_transform(df_ml['tournament'])

# We need dummy values for FIFA ratings / xG since we simplified the pipeline
# We fill these with neutral values so the array shape matches the 34 features expected by worldcup.py
df_ml['h_rat'] = 65.0
df_ml['a_rat'] = 65.0
df_ml['h_xg'] = 1.2
df_ml['a_xg'] = 1.2

# Compute diffs and date features first
df_ml['year'] = df_ml['date'].dt.year
df_ml['month'] = df_ml['date'].dt.month
df_ml['elo_diff'] = df_ml['home_elo'] - df_ml['away_elo']
df_ml['form_diff'] = df_ml['h_form'] - df_ml['a_form']
df_ml['rat_diff'] = df_ml['h_rat'] - df_ml['a_rat']
df_ml['xg_diff'] = df_ml['h_xg'] - df_ml['a_xg']
df_ml['gd_diff'] = df_ml['h_gd'] - df_ml['a_gd']

df_ml['h_comp'] = df_ml['h_form']
df_ml['a_comp'] = df_ml['a_form']
df_ml['comp_diff'] = df_ml['form_diff']

df_ml['h_form2'] = df_ml['h_form']
df_ml['a_form2'] = df_ml['a_form']
df_ml['form_diff2'] = df_ml['form_diff']

df_ml['h_xga'] = df_ml['h_xg']
df_ml['a_xga'] = df_ml['a_xg']
df_ml['xga_diff'] = df_ml['xg_diff']

# 34 columns to match worldcup.py expectation
feature_cols = [
    'home_enc', 'away_enc', 'tour_enc', 'tour_tier', 
    'year', 'month', 'neutral',
    'home_elo', 'away_elo', 'elo_diff',
    'h_form', 'a_form', 'form_diff',
    'h_comp', 'a_comp', 'comp_diff',
    'h_form2', 'a_form2', 'form_diff2',
    'h_rat', 'a_rat', 'rat_diff',
    'h_xg', 'a_xg', 'xg_diff',
    'h_xga', 'a_xga', 'xga_diff',
    'h_gd', 'a_gd', 'gd_diff',
    'h2h_hw', 'h2h_aw', 'h2h_d'
]

df_ml = df_ml.fillna(0)
X = df_ml[feature_cols]
y = df_ml['result']

# Train/Test Split (Time-based: train on past, test on latest)
split_idx = int(len(X) * 0.9)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ─────────────────────────────────────────────────────────────
# 7. TRAIN ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────
print("🚀 Training ensemble (XGBoost + RandomForest + LogisticRegression)...")

xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, eval_metric='mlogloss')
rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)

ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
    voting='soft',
    weights=[2, 1.5, 1]
)

# Encode targets for XGBoost
le_y = LabelEncoder()
y_train_enc = le_y.fit_transform(y_train)
y_test_enc = le_y.transform(y_test)

ensemble.fit(X_train, y_train_enc)
preds = ensemble.predict(X_test)
acc = accuracy_score(y_test_enc, preds)

print("="*60)
print(f"  ENSEMBLE ACCURACY: {acc*100:.2f}%")
print("="*60)

# ─────────────────────────────────────────────────────────────
# 8. EXPORT ARTIFACTS
# ─────────────────────────────────────────────────────────────
# Retrain on ALL data for production
ensemble.fit(X, le_y.fit_transform(y))

# Build lookup dictionaries for worldcup.py to use in live predictions
latest_stats = {}
for team in df_results['home_team'].unique():
    hs = team_stats.get(team, {'history':[], 'gd':[]})
    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    h_gd = np.mean(hs['gd'][-10:]) if hs['gd'] else 0.0
    latest_stats[team] = {
        'elo': elo_dict.get(team, 1500),
        'win_rate': h_form,
        'recent_form': h_form,
        'comp_recent': h_form,
        'gd_avg': h_gd,
        'rolling_xg': 1.2,
        'rolling_xga': 1.2
    }

# Convert h2h dictionary format for worldcup.py
# worldcup.py expects a list of dicts [{'winner': home}, ...]
h2h_export = {}
for pair, hist in h2h_history.items():
    h2h_export[pair] = [{'winner': w if w != 'Draw' else 'draw'} for w in hist]

encoders = {
    'le_home': le_home,
    'le_away': le_away,
    'le_tour': le_tour,
    'team_rating': {},  # Simplified
    'latest_stats': latest_stats,
    'h2h_history': h2h_export,
    'elo_ratings': elo_dict,
    'features': feature_cols,
    'accuracy': round(acc * 100, 2)
}

with open('football_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Model & encoders successfully saved!")