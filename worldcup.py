import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request
import pickle
import json

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL & METADATA
# ─────────────────────────────────────────────────────────────
with open('football_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    enc = pickle.load(f)

le_home      = enc['le_home']
le_away      = enc['le_away']
le_tour      = enc['le_tour']
team_rating  = enc['team_rating']
latest_stats = enc['latest_stats']
h2h_history  = enc['h2h_history']
elo_ratings  = enc['elo_ratings']
FEATURES     = enc['features']
MODEL_ACC    = enc['accuracy']
ELO_START    = 1500
DEFAULT_RATING = 65.0
DEFAULT_STATS  = {'win_rate': 0.45, 'recent_form': 0.45, 'comp_recent': 0.45,
                   'rolling_xg': 1.2, 'rolling_xga': 1.2, 'gd_avg': 0.0, 'elo': ELO_START}

# ─────────────────────────────────────────────────────────────
# WORLD CUP 2026 DATA
# ─────────────────────────────────────────────────────────────
FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Czech Republic": "🇨🇿", "Spain": "🇪🇸", "Croatia": "🇭🇷",
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "France": "🇫🇷",
    "Belgium": "🇧🇪", "Austria": "🇦🇹", "Brazil": "🇧🇷",
    "Germany": "🇩🇪", "Netherlands": "🇳🇱", "Norway": "🇳🇴",
    "Portugal": "🇵🇹", "Colombia": "🇨🇴", "United States": "🇺🇸",
    "Paraguay": "🇵🇾", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Japan": "🇯🇵",
    "Ecuador": "🇪🇨", "Uruguay": "🇺🇾", "Switzerland": "🇨🇭",
    "Turkey": "🇹🇷", "Ghana": "🇬🇭", "Senegal": "🇸🇳",
    "Morocco": "🇲🇦", "Australia": "🇦🇺", "Canada": "🇨🇦",
    "Sweden": "🇸🇪", "Tunisia": "🇹🇳", "Italy": "🇮🇹",
    "Iran": "🇮🇷", "Saudi Arabia": "🇸🇦", "Poland": "🇵🇱",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Denmark": "🇩🇰", "Serbia": "🇷🇸",
    "Cameroon": "🇨🇲", "Costa Rica": "🇨🇷", "Qatar": "🇶🇦",
    "South Korea": "🇰🇷", "Ghana": "🇬🇭", "Nigeria": "🇳🇬",
    "Egypt": "🇪🇬", "Chile": "🇨🇱", "Peru": "🇵🇪",
    "New Zealand": "🇳🇿", "Indonesia": "🇮🇩",
}

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Spain", "Croatia", "Algeria", "Argentina"],
    "C": ["France", "Belgium", "Austria", "Brazil"],
    "D": ["Germany", "Netherlands", "Norway", "Portugal"],
    "E": ["Colombia", "United States", "Paraguay", "England"],
    "F": ["Japan", "Ecuador", "Uruguay", "Switzerland"],
    "G": ["Turkey", "Ghana", "Senegal", "Morocco"],
    "H": ["Australia", "Canada", "Sweden", "Tunisia"]
}

ALL_TEAMS = sorted(FLAGS.keys())

# ─────────────────────────────────────────────────────────────
# PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────
TOURNAMENT_TIER = {
    'FIFA World Cup': 3.0, 'Copa América': 2.5, 'UEFA Euro': 2.5,
    'Africa Cup of Nations': 2.0, 'AFC Asian Cup': 2.0,
    'CONCACAF Gold Cup': 2.0, 'UEFA Nations League': 1.8,
    'FIFA World Cup qualification': 1.5, 'Friendly': 0.7,
}

def predict(home, away, tournament="FIFA World Cup", year=2026, month=6, neutral=1):
    try:
        known_homes = list(le_home.classes_)
        known_aways = list(le_away.classes_)
        known_tours = list(le_tour.classes_)
        h_enc = le_home.transform([home])[0] if home in known_homes else 0
        a_enc = le_away.transform([away])[0] if away in known_aways else 0
        t_enc = le_tour.transform([tournament])[0] if tournament in known_tours else 0
        tour_tier = 3.0

        hs  = latest_stats.get(home, DEFAULT_STATS)
        as_ = latest_stats.get(away, DEFAULT_STATS)

        h_elo  = hs.get('elo', ELO_START);      a_elo  = as_.get('elo', ELO_START)
        h_wr   = hs.get('win_rate', 0.45);      a_wr   = as_.get('win_rate', 0.45)
        h_rec  = hs.get('recent_form', 0.45);   a_rec  = as_.get('recent_form', 0.45)
        h_crec = hs.get('comp_recent', 0.45);   a_crec = as_.get('comp_recent', 0.45)
        h_rat  = team_rating.get(home, DEFAULT_RATING)
        a_rat  = team_rating.get(away, DEFAULT_RATING)
        h_xg   = hs.get('rolling_xg', 1.2);     a_xg   = as_.get('rolling_xg', 1.2)
        h_xga  = hs.get('rolling_xga', 1.2);    a_xga  = as_.get('rolling_xga', 1.2)
        h_gd   = hs.get('gd_avg', 0.0);         a_gd   = as_.get('gd_avg', 0.0)

        h2h_key  = tuple(sorted([home, away]))
        h2h_hist = h2h_history.get(h2h_key, [])
        if isinstance(h2h_hist, list):
            h2h_hist = h2h_hist[-5:]
        h2h_hw = sum(1 for m in h2h_hist if m.get('winner') == home)
        h2h_aw = sum(1 for m in h2h_hist if m.get('winner') == away)
        h2h_d  = sum(1 for m in h2h_hist if m.get('winner') == 'draw')

        features = [
            h_enc, a_enc, t_enc, tour_tier, year, month, neutral,
            h_elo, a_elo, h_elo - a_elo,
            h_wr, a_wr, h_wr - a_wr,
            h_rec, a_rec, h_rec - a_rec,
            h_crec, a_crec, h_crec - a_crec,
            h_rat, a_rat, h_rat - a_rat,
            h_xg, a_xg, h_xg - a_xg,
            h_xga, a_xga, h_xga - a_xga,
            h_gd, a_gd, h_gd - a_gd,
            h2h_hw, h2h_aw, h2h_d,
        ]

        proba = model.predict_proba([features])[0]
        elo_diff = abs(h_elo - a_elo)
        confidence = "High" if elo_diff > 200 else ("Medium" if elo_diff > 80 else "Low")

        return {
            'home': round(proba[0]*100, 1),
            'away': round(proba[1]*100, 1),
            'draw': round(proba[2]*100, 1),
            'home_elo': round(h_elo),
            'away_elo': round(a_elo),
            'home_rating': round(h_rat, 1),
            'away_rating': round(a_rat, 1),
            'home_xg': round(h_xg, 2),
            'away_xg': round(a_xg, 2),
            'h2h_home': h2h_hw,
            'h2h_away': h2h_aw,
            'h2h_draw': h2h_d,
            'confidence': confidence,
        }
    except Exception as e:
        print(f"Prediction error ({home} vs {away}): {e}")
        return {'home': 33.3, 'away': 33.3, 'draw': 33.3,
                'home_elo': ELO_START, 'away_elo': ELO_START,
                'home_rating': DEFAULT_RATING, 'away_rating': DEFAULT_RATING,
                'home_xg': 1.2, 'away_xg': 1.2, 'h2h_home': 0, 'h2h_away': 0,
                'h2h_draw': 0, 'confidence': 'Low'}

# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE STANDINGS & FIXTURES
# ─────────────────────────────────────────────────────────────
all_teams = [t for teams in GROUPS.values() for t in teams]

champ_data = []
for team in all_teams:
    avg_win = sum(predict(team, t2)['home'] for t2 in all_teams if t2 != team) / (len(all_teams) - 1)
    champ_data.append({
        'team': team,
        'champ': round(avg_win, 1),
        'elo': round(elo_ratings.get(team, ELO_START)),
        'rating': round(team_rating.get(team, DEFAULT_RATING), 1),
    })
champ_data.sort(key=lambda x: x['champ'], reverse=True)

fixtures = []
for grp, teams in GROUPS.items():
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            p = predict(teams[i], teams[j])
            fixtures.append({
                'home_team': teams[i], 'away_team': teams[j],
                'group': grp, **p
            })

def flag(team): return FLAGS.get(team, "🏳️")

# ─────────────────────────────────────────────────────────────
# API ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route('/api/predict')
def api_predict_route():
    home = request.args.get('home', '')
    away = request.args.get('away', '')
    if not home or not away or home == away:
        return jsonify({'error': 'Invalid teams'}), 400
    result = predict(home, away)
    result['home_flag'] = flag(home)
    result['away_flag'] = flag(away)
    result['home_name'] = home
    result['away_name'] = away
    return jsonify(result)

# ─────────────────────────────────────────────────────────────
# MAIN ROUTE — Full Dashboard
# ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    # Build team options for predictor dropdown
    team_options = ""
    for t in sorted(set(list(FLAGS.keys()) + all_teams)):
        team_options += f'<option value="{t}">{flag(t)} {t}</option>'

    # Standings rows
    standings_rows = ""
    for i, s in enumerate(champ_data):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}"
        width = s['champ']
        bar_color = ["#f59e0b", "#9ca3af", "#b45309"][i] if i < 3 else "#7c3aed"
        standings_rows += f"""
        <tr class="standings-row" style="animation-delay:{i*0.04}s">
            <td class="rank-cell">{medal}</td>
            <td class="team-cell"><span class="team-flag">{flag(s['team'])}</span><span class="team-name">{s['team']}</span></td>
            <td><span class="elo-badge">{s['elo']}</span></td>
            <td><span class="rating-badge">{s['rating']}</span></td>
            <td class="prob-cell">
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{min(width*2,100)}%;background:{bar_color}"></div>
                </div>
                <span class="prob-val">{s['champ']}%</span>
            </td>
        </tr>"""

    # Fixture cards
    fixture_cards = ""
    for f in fixtures:
        hc = f['home']; ac = f['away']; dc = f['draw']
        conf_color = {"High": "#10b981", "Medium": "#f59e0b", "Low": "#6b7280"}[f['confidence']]
        ht = f['home_team']; at = f['away_team']
        fixture_cards += f"""
        <div class="fixture-card">
            <div class="fixture-header">
                <span class="fixture-group">Group {f['group']}</span>
                <span class="confidence-badge" style="color:{conf_color}">● {f['confidence']} Confidence</span>
            </div>
            <div class="fixture-teams">
                <div class="fixture-team">
                    <div class="fixture-flag-sm">{flag(ht)}</div>
                    <div class="fixture-name-big">{ht}</div>
                    <div class="fixture-prob-num" style="color:var(--purple-light)">{hc}%</div>
                </div>
                <div class="fixture-vs-col">
                    <div class="fixture-vs-box">VS</div>
                    <div class="fixture-draw-pct">{dc}% Draw</div>
                </div>
                <div class="fixture-team fixture-team-r">
                    <div class="fixture-flag-sm">{flag(at)}</div>
                    <div class="fixture-name-big">{at}</div>
                    <div class="fixture-prob-num" style="color:var(--pink)">{ac}%</div>
                </div>
            </div>
            <div class="prob-strip">
                <div class="prob-seg prob-home" style="flex:{hc}"></div>
                <div class="prob-seg prob-draw" style="flex:{dc}"></div>
                <div class="prob-seg prob-away" style="flex:{ac}"></div>
            </div>
        </div>"""

    # Groups grid
    groups_html = '<div class="groups-grid">'
    for g, ts in GROUPS.items():
        groups_html += f'<div class="group-card"><div class="group-label">Group {g}</div>'
        for t in ts:
            elo = round(elo_ratings.get(t, ELO_START))
            groups_html += f"""
            <div class="group-team">
                <span class="group-flag">{flag(t)}</span>
                <span class="group-name">{t}</span>
                <span class="group-elo">ELO {elo}</span>
            </div>"""
        groups_html += '</div>'
    groups_html += '</div>'

    # Champion prediction
    winner = champ_data[0]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>World Cup 2026 — Predictions</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #080b14;
            --bg2: #0d1120;
            --bg3: #111827;
            --glass: rgba(255,255,255,0.04);
            --glass-border: rgba(255,255,255,0.08);
            --purple: #7c3aed;
            --purple-light: #a855f7;
            --pink: #ec4899;
            --gold: #f59e0b;
            --green: #10b981;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --text-dim: #475569;
        }}
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        html {{ scroll-behavior: smooth; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* ── HEADER ── */
        .header {{
            position: relative;
            padding: 50px 20px 40px;
            text-align: center;
            background: linear-gradient(135deg, #0f0520 0%, #1a0533 40%, #0f172a 100%);
            overflow: hidden;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: -50%; left: 50%;
            transform: translateX(-50%);
            width: 600px; height: 600px;
            background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
            pointer-events: none;
        }}
        .header::after {{
            content: '';
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(124,58,237,0.5), rgba(236,72,153,0.5), transparent);
        }}
        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(124,58,237,0.15);
            border: 1px solid rgba(124,58,237,0.3);
            border-radius: 100px;
            padding: 6px 16px;
            font-size: 0.75em;
            font-weight: 600;
            color: var(--purple-light);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }}
        .header h1 {{
            font-size: clamp(2em, 5vw, 3.5em);
            font-weight: 900;
            background: linear-gradient(135deg, #fff 30%, var(--purple-light) 70%, var(--pink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
            line-height: 1.1;
            margin-bottom: 12px;
        }}
        .header-sub {{
            color: var(--text-muted);
            font-size: 0.95em;
            font-weight: 400;
            margin-bottom: 24px;
        }}
        .acc-pills {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .acc-pill {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 100px;
            padding: 8px 20px;
            font-size: 0.82em;
            font-weight: 600;
        }}
        .acc-pill span {{ color: var(--gold); }}

        /* ── TABS ── */
        .tabs-wrap {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(8,11,20,0.9);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
        }}
        .tabs {{
            display: flex;
            max-width: 1100px;
            margin: 0 auto;
            padding: 0 20px;
            overflow-x: auto;
            scrollbar-width: none;
        }}
        .tabs::-webkit-scrollbar {{ display: none; }}
        .tab {{
            flex-shrink: 0;
            padding: 16px 22px;
            font-size: 0.85em;
            font-weight: 600;
            color: var(--text-dim);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab:hover {{ color: var(--text-muted); }}
        .tab.active {{
            color: var(--purple-light);
            border-bottom-color: var(--purple);
        }}

        /* ── PAGE CONTENT ── */
        .page {{ display: none; padding: 32px 20px 60px; max-width: 1100px; margin: 0 auto; }}
        .page.active {{ display: block; animation: fadeUp 0.3s ease; }}
        @keyframes fadeUp {{
            from {{ opacity:0; transform: translateY(12px); }}
            to   {{ opacity:1; transform: translateY(0); }}
        }}
        .section-title {{
            font-size: 1.4em;
            font-weight: 800;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: var(--glass-border);
            margin-left: 12px;
        }}

        /* ── STANDINGS TABLE ── */
        .standings-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            overflow: hidden;
        }}
        .standings-table th {{
            padding: 14px 16px;
            text-align: left;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            background: rgba(255,255,255,0.02);
            border-bottom: 1px solid var(--glass-border);
        }}
        .standings-row {{
            border-bottom: 1px solid rgba(255,255,255,0.04);
            transition: background 0.2s;
            animation: fadeUp 0.4s ease backwards;
        }}
        .standings-row:hover {{ background: rgba(124,58,237,0.08); }}
        .standings-row td {{ padding: 14px 16px; font-size: 0.9em; vertical-align: middle; }}
        .rank-cell {{ font-size: 1.1em; width: 48px; }}
        .team-cell {{ display: flex; align-items: center; gap: 10px; }}
        .team-flag {{ font-size: 1.4em; }}
        .team-name {{ font-weight: 600; }}
        .elo-badge {{
            background: rgba(124,58,237,0.15);
            color: var(--purple-light);
            border: 1px solid rgba(124,58,237,0.2);
            padding: 3px 10px; border-radius: 100px; font-size: 0.8em; font-weight: 600;
        }}
        .rating-badge {{
            background: rgba(245,158,11,0.12);
            color: var(--gold);
            border: 1px solid rgba(245,158,11,0.2);
            padding: 3px 10px; border-radius: 100px; font-size: 0.8em; font-weight: 600;
        }}
        .prob-cell {{ display: flex; align-items: center; gap: 12px; min-width: 160px; }}
        .prob-bar-wrap {{
            flex: 1; height: 6px; background: rgba(255,255,255,0.06);
            border-radius: 100px; overflow: hidden;
        }}
        .prob-bar-fill {{ height: 100%; border-radius: 100px; transition: width 1s ease; }}
        .prob-val {{ font-weight: 700; font-size: 0.9em; color: var(--text-muted); min-width: 42px; }}

        /* ── FIXTURE CARDS ── */
        .fixtures-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 16px;
        }}
        .fixture-card {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.25s;
        }}
        .fixture-card:hover {{
            border-color: rgba(124,58,237,0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(124,58,237,0.12);
        }}
        .fixture-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .fixture-group {{
            font-size: 0.72em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--purple-light);
            background: rgba(124,58,237,0.12);
            border: 1px solid rgba(124,58,237,0.2);
            padding: 3px 10px;
            border-radius: 100px;
        }}
        .confidence-badge {{ font-size: 0.75em; font-weight: 600; }}
        .fixture-teams {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }}
        .fixture-team {{ display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }}
        .fixture-team-r {{ align-items: flex-end; text-align: right; }}
        .fixture-flag-sm {{ font-size: 1.6em; line-height: 1; }}
        .fixture-name-big {{ font-weight: 800; font-size: 1.05em; line-height: 1.2; text-transform: uppercase; letter-spacing: -0.5px; }}
        .fixture-prob-num {{ font-size: 1.4em; font-weight: 900; line-height: 1; margin-top: 2px; }}
        .fixture-vs-col {{ display: flex; flex-direction: column; align-items: center; gap: 8px; justify-content: center; }}
        .fixture-vs-box {{
            font-size: 0.7em; font-weight: 800;
            color: var(--text-dim);
            background: rgba(255,255,255,0.06);
            padding: 4px 10px;
            border-radius: 8px;
        }}
        .fixture-draw-pct {{ font-size: 0.75em; color: var(--text-muted); }}
        .prob-strip {{
            display: flex;
            height: 10px;
            border-radius: 8px;
            overflow: hidden;
            gap: 2px;
            margin-bottom: 8px;
        }}
        .prob-seg {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0;
            transition: flex 0.8s ease;
            border-radius: 4px;
            min-width: 2px;
        }}
        .prob-home {{ background: linear-gradient(90deg, var(--purple), var(--purple-light)); }}
        .prob-draw {{ background: rgba(255,255,255,0.15); }}
        .prob-away {{ background: linear-gradient(90deg, var(--pink), #f472b6); }}
        .prob-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.74em;
            color: var(--text-dim);
            font-weight: 500;
        }}

        /* ── GROUPS GRID ── */
        .groups-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 16px;
        }}
        .group-card {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
        }}
        .group-label {{
            font-size: 0.75em;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--purple-light);
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--glass-border);
        }}
        .group-team {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}
        .group-team:last-child {{ border-bottom: none; }}
        .group-flag {{ font-size: 1.3em; }}
        .group-name {{ flex: 1; font-weight: 500; font-size: 0.9em; }}
        .group-elo {{
            font-size: 0.72em;
            font-weight: 600;
            color: var(--text-dim);
            background: rgba(255,255,255,0.05);
            padding: 2px 8px;
            border-radius: 6px;
        }}

        /* ── CHAMPION CARD ── */
        .champion-card {{
            background: linear-gradient(135deg, #1a0533 0%, rgba(124,58,237,0.3) 50%, rgba(236,72,153,0.2) 100%);
            border: 1px solid rgba(124,58,237,0.4);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 32px;
            position: relative;
            overflow: hidden;
        }}
        .champion-card::before {{
            content: '';
            position: absolute;
            top: -50%; left: 50%;
            transform: translateX(-50%);
            width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%);
            pointer-events: none;
        }}
        .champion-flag {{ font-size: 5em; margin-bottom: 12px; display: block; }}
        .champion-title {{
            font-size: 0.75em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--gold);
            margin-bottom: 8px;
        }}
        .champion-name {{
            font-size: 2.5em;
            font-weight: 900;
            background: linear-gradient(135deg, #fff, var(--purple-light));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}
        .champion-prob {{
            color: var(--text-muted);
            font-size: 0.95em;
        }}

        /* ── PREDICTOR ── */
        .predictor-card {{
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 32px;
        }}
        .predictor-teams {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 16px;
            align-items: end;
            margin-bottom: 24px;
        }}
        .select-wrap {{ display: flex; flex-direction: column; gap: 8px; }}
        .select-label {{ font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); }}
        .team-select {{
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 14px 16px;
            color: var(--text);
            font-family: inherit;
            font-size: 0.95em;
            font-weight: 500;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s;
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
        }}
        .team-select:focus {{ border-color: var(--purple); }}
        .vs-label {{
            font-size: 0.8em;
            font-weight: 800;
            color: var(--text-dim);
            text-align: center;
            padding: 14px 0;
        }}
        .predict-btn {{
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, var(--purple), var(--pink));
            border: none;
            border-radius: 12px;
            color: white;
            font-family: inherit;
            font-size: 1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            letter-spacing: 0.3px;
        }}
        .predict-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 24px rgba(124,58,237,0.4);
        }}
        .predict-btn:active {{ transform: translateY(0); }}

        /* Result panel */
        .result-panel {{
            display: none;
            margin-top: 28px;
            animation: fadeUp 0.35s ease;
        }}
        .result-panel.visible {{ display: block; }}
        .result-teams-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        .result-team {{
            text-align: center;
        }}
        .result-flag {{ font-size: 3em; display: block; margin-bottom: 6px; }}
        .result-name {{ font-weight: 700; font-size: 1em; }}
        .result-vs {{ font-size: 1.5em; font-weight: 900; color: var(--text-dim); }}
        .result-probs {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}
        .result-prob-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
        }}
        .result-prob-card.winner {{ border-color: rgba(124,58,237,0.4); background: rgba(124,58,237,0.08); }}
        .result-prob-label {{ font-size: 0.72em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); margin-bottom: 8px; }}
        .result-prob-val {{ font-size: 2em; font-weight: 900; }}
        .result-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 10px;
        }}
        .result-stat-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 14px;
        }}
        .stat-label {{ font-size: 0.7em; color: var(--text-dim); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .stat-vals {{ display: flex; justify-content: space-between; font-weight: 700; font-size: 0.9em; }}
        .stat-home {{ color: var(--purple-light); }}
        .stat-away {{ color: var(--pink); }}
        .confidence-row {{ text-align: center; margin-top: 16px; }}
        .conf-chip {{
            display: inline-block;
            padding: 6px 18px;
            border-radius: 100px;
            font-size: 0.8em;
            font-weight: 700;
        }}
        .conf-High {{ background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }}
        .conf-Medium {{ background: rgba(245,158,11,0.12); color: var(--gold); border: 1px solid rgba(245,158,11,0.25); }}
        .conf-Low {{ background: rgba(107,114,128,0.12); color: #9ca3af; border: 1px solid rgba(107,114,128,0.2); }}

        /* ── LOADING SPINNER ── */
        .spinner {{
            width: 40px; height: 40px;
            border: 3px solid rgba(124,58,237,0.2);
            border-top-color: var(--purple);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
            margin: 30px auto;
            display: none;
        }}
        .spinner.active {{ display: block; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        /* ── SCROLLBAR ── */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg2); }}
        ::-webkit-scrollbar-thumb {{ background: rgba(124,58,237,0.4); border-radius: 3px; }}

        @media (max-width: 640px) {{
            .predictor-teams {{ grid-template-columns: 1fr; }}
            .standings-table th:nth-child(3),
            .standings-table td:nth-child(3) {{ display: none; }}
            .fixtures-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="header-badge">⚽ Data Driven · World Cup 2026</div>
    <h1>World Cup 2026<br>Predictions</h1>
    <p class="header-sub">XGBoost + Random Forest + Logistic Regression Ensemble · Strength-Adjusted Rolling xG · Elo Ratings</p>
    <div class="acc-pills">
        <div class="acc-pill">Model Accuracy <span>{MODEL_ACC}%</span></div>
        <div class="acc-pill">Features <span>34</span></div>
        <div class="acc-pill">Training Matches <span>13,888</span></div>
        <div class="acc-pill">Ensemble Models <span>3</span></div>
    </div>
</div>

<!-- TABS -->
<div class="tabs-wrap">
    <div class="tabs">
        <div class="tab active" onclick="showTab('standings',this)">🏆 Standings</div>
        <div class="tab" onclick="showTab('fixtures',this)">📅 Fixtures</div>
        <div class="tab" onclick="showTab('groups',this)">👥 Groups</div>
        <div class="tab" onclick="showTab('bracket',this)">🎯 Bracket</div>
        <div class="tab" onclick="showTab('predictor',this)">⚡ Predict Match</div>
    </div>
</div>

<!-- STANDINGS -->
<div id="standings" class="page active">
    <div class="section-title">🏆 Championship Predictions</div>
    <table class="standings-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Team</th>
                <th>Elo</th>
                <th>FIFA Rating</th>
                <th>Win Probability</th>
            </tr>
        </thead>
        <tbody>{standings_rows}</tbody>
    </table>
</div>

<!-- FIXTURES -->
<div id="fixtures" class="page">
    <div class="section-title">📅 Group Stage Fixtures</div>
    <div class="fixtures-grid">{fixture_cards}</div>
</div>

<!-- GROUPS -->
<div id="groups" class="page">
    <div class="section-title">👥 Groups</div>
    {groups_html}
</div>

<!-- BRACKET -->
<div id="bracket" class="page">
    <div class="section-title">🎯 Predicted Champion</div>
    <div class="champion-card">
        <span class="champion-flag">{flag(winner['team'])}</span>
        <div class="champion-title">🏆 Most Likely Winner</div>
        <div class="champion-name">{winner['team']}</div>
        <div class="champion-prob">{winner['champ']}% average win probability in group stage simulations</div>
    </div>
    <div class="section-title">Top 8 Contenders</div>
    <div class="groups-grid">
        {''.join(f"""
        <div class="group-card" style="border-color:rgba(124,58,237,{0.4 if i==0 else 0.15})">
            <div class="group-label" style="color:{'#f59e0b' if i==0 else 'var(--purple-light)'}">
                {'🥇 Favourite' if i==0 else f'#{i+1} Contender'}
            </div>
            <div class="group-team">
                <span class="group-flag">{flag(s['team'])}</span>
                <span class="group-name">{s['team']}</span>
                <span class="group-elo" style="color:var(--purple-light)">{s['champ']}%</span>
            </div>
        </div>""" for i, s in enumerate(champ_data[:8]))}
    </div>
</div>

<!-- PREDICTOR -->
<div id="predictor" class="page">
    <div class="section-title">⚡ Predict Any Match</div>
    <div class="predictor-card">
        <div class="predictor-teams">
            <div class="select-wrap">
                <label class="select-label">Home Team</label>
                <select id="home-team" class="team-select">
                    <option value="">Select team...</option>
                    {team_options}
                </select>
            </div>
            <div class="vs-label">VS</div>
            <div class="select-wrap">
                <label class="select-label">Away Team</label>
                <select id="away-team" class="team-select">
                    <option value="">Select team...</option>
                    {team_options}
                </select>
            </div>
        </div>
        <button class="predict-btn" onclick="runPredict()">⚡ Generate Prediction</button>

        <div class="spinner" id="spinner"></div>

        <div class="result-panel" id="result-panel">
            <div class="result-teams-row">
                <div class="result-team">
                    <span class="result-flag" id="res-home-flag"></span>
                    <div class="result-name" id="res-home-name"></div>
                </div>
                <div class="result-vs">VS</div>
                <div class="result-team" style="text-align:right">
                    <span class="result-flag" id="res-away-flag"></span>
                    <div class="result-name" id="res-away-name"></div>
                </div>
            </div>
            <div class="result-probs">
                <div class="result-prob-card" id="card-home">
                    <div class="result-prob-label" id="label-home">Home Win</div>
                    <div class="result-prob-val" id="val-home" style="color:var(--purple-light)"></div>
                </div>
                <div class="result-prob-card">
                    <div class="result-prob-label">Draw</div>
                    <div class="result-prob-val" id="val-draw" style="color:var(--text-muted)"></div>
                </div>
                <div class="result-prob-card" id="card-away">
                    <div class="result-prob-label" id="label-away">Away Win</div>
                    <div class="result-prob-val" id="val-away" style="color:var(--pink)"></div>
                </div>
            </div>
            <div class="result-stats">
                <div class="result-stat-card">
                    <div class="stat-label">Elo Rating</div>
                    <div class="stat-vals">
                        <span class="stat-home" id="stat-elo-h"></span>
                        <span class="stat-away" id="stat-elo-a"></span>
                    </div>
                </div>
                <div class="result-stat-card">
                    <div class="stat-label">FIFA Rating</div>
                    <div class="stat-vals">
                        <span class="stat-home" id="stat-rat-h"></span>
                        <span class="stat-away" id="stat-rat-a"></span>
                    </div>
                </div>
                <div class="result-stat-card">
                    <div class="stat-label">Rolling xG</div>
                    <div class="stat-vals">
                        <span class="stat-home" id="stat-xg-h"></span>
                        <span class="stat-away" id="stat-xg-a"></span>
                    </div>
                </div>
                <div class="result-stat-card">
                    <div class="stat-label">H2H (last 5)</div>
                    <div class="stat-vals">
                        <span class="stat-home" id="stat-h2h-h"></span>
                        <span class="stat-away" id="stat-h2h-a"></span>
                    </div>
                </div>
            </div>
            <div class="confidence-row">
                <span class="conf-chip" id="conf-chip"></span>
            </div>
        </div>
    </div>
</div>

<script>
    function showTab(id, el) {{
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        el.classList.add('active');
    }}

    async function runPredict() {{
        const home = document.getElementById('home-team').value;
        const away = document.getElementById('away-team').value;
        if (!home || !away) {{ alert('Please select both teams!'); return; }}
        if (home === away) {{ alert('Teams must be different!'); return; }}

        document.getElementById('result-panel').classList.remove('visible');
        document.getElementById('spinner').classList.add('active');

        try {{
            const res = await fetch(`/api/predict?home=${{encodeURIComponent(home)}}&away=${{encodeURIComponent(away)}}`);
            const d = await res.json();
            document.getElementById('spinner').classList.remove('active');

            document.getElementById('res-home-flag').textContent = d.home_flag;
            document.getElementById('res-away-flag').textContent = d.away_flag;
            document.getElementById('res-home-name').textContent = d.home_name;
            document.getElementById('res-away-name').textContent = d.away_name;
            document.getElementById('label-home').textContent = d.home_name + ' Win';
            document.getElementById('label-away').textContent = d.away_name + ' Win';

            const hv = d.home, av = d.away, dv = d.draw;
            document.getElementById('val-home').textContent = hv + '%';
            document.getElementById('val-away').textContent = av + '%';
            document.getElementById('val-draw').textContent = dv + '%';

            // Highlight winner
            document.getElementById('card-home').classList.remove('winner');
            document.getElementById('card-away').classList.remove('winner');
            if (hv > av && hv > dv) document.getElementById('card-home').classList.add('winner');
            else if (av > hv && av > dv) document.getElementById('card-away').classList.add('winner');

            document.getElementById('stat-elo-h').textContent = d.home_elo;
            document.getElementById('stat-elo-a').textContent = d.away_elo;
            document.getElementById('stat-rat-h').textContent = d.home_rating;
            document.getElementById('stat-rat-a').textContent = d.away_rating;
            document.getElementById('stat-xg-h').textContent = d.home_xg;
            document.getElementById('stat-xg-a').textContent = d.away_xg;
            document.getElementById('stat-h2h-h').textContent = d.h2h_home + 'W';
            document.getElementById('stat-h2h-a').textContent = d.h2h_away + 'W';

            const chip = document.getElementById('conf-chip');
            chip.textContent = '● ' + d.confidence + ' Confidence Prediction';
            chip.className = 'conf-chip conf-' + d.confidence;

            document.getElementById('result-panel').classList.add('visible');
        }} catch(e) {{
            document.getElementById('spinner').classList.remove('active');
            alert('Prediction failed. Please try again.');
        }}
    }}

    // Allow pressing Enter on selects
    document.addEventListener('keydown', e => {{
        if (e.key === 'Enter' && document.getElementById('predictor').classList.contains('active')) runPredict();
    }});
</script>
</body>
</html>"""

if __name__ == '__main__':
    print(f"\n⚽ World Cup 2026 Predictor")
    print(f"   Model Accuracy: {MODEL_ACC}%  |  Ensemble: XGBoost + RF + LR")
    print(f"   Dashboard: http://127.0.0.1:5001\n")
    app.run(debug=True, port=5001)