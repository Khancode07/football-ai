from flask import Flask
import pickle

app = Flask(__name__)

with open('football_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    enc = pickle.load(f)

le_home = enc['le_home']
le_away = enc['le_away']
win_rate = enc['win_rate']

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
    "Sweden": "🇸🇪", "Tunisia": "🇹🇳"
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

def predict(home, away):
    try:
        h = le_home.transform([home])[0]
        a = le_away.transform([away])[0]
        hw = win_rate.get(home, 0.5)
        aw = win_rate.get(away, 0.5)
        wd = hw - aw
        proba = model.predict_proba([[h, a, 0, 2026, 6, hw, aw, wd, hw, aw, wd]])[0]
        return round(proba[0]*100,1), round(proba[1]*100,1), round(proba[2]*100,1)
    except:
        return 33.3, 33.3, 33.3

all_teams = [t for teams in GROUPS.values() for t in teams]
champ = []
for team in all_teams:
    wins = 0
    for t2 in all_teams:
        if t2 != team:
            h, a, d = predict(team, t2)
            wins += h
    cp = round(wins/(len(all_teams)-1), 1)
    champ.append({"team": team, "champ": cp, "final": round(cp*1.8,1)})
champ.sort(key=lambda x: x['champ'], reverse=True)

fixtures = []
for grp, teams in GROUPS.items():
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            h, a, d = predict(teams[i], teams[j])
            fixtures.append({
                "home": teams[i], "away": teams[j],
                "group": grp, "hp": h, "ap": a, "dp": d
            })

def flag(team):
    return FLAGS.get(team, "🏳️")

@app.route('/')
def index():
    # Standings
    rows = ""
    for i, s in enumerate(champ):
        rows += "<tr>"
        rows += "<td style='color:#ec4899;font-weight:bold;font-size:1.2em'>" + str(i+1).zfill(2) + "</td>"
        rows += "<td>" + flag(s['team']) + " " + s['team'] + "</td>"
        rows += "<td>" + str(s['final']) + "%</td>"
        rows += "<td><span style='background:#fce7f3;color:#ec4899;padding:4px 12px;border-radius:20px;font-weight:bold'>" + str(s['champ']) + "%</span></td>"
        rows += "</tr>"

    # Fixtures
    fix = ""
    for f in fixtures:
        fix += "<div style='background:white;border-radius:12px;padding:15px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        fix += "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
        fix += "<div style='font-weight:bold;font-size:1.1em'>" + flag(f['home']) + " " + f['home'] + "</div>"
        fix += "<div style='color:#888;font-size:0.85em;text-align:center'>Group " + f['group'] + "</div>"
        fix += "<div style='font-weight:bold;font-size:1.1em;text-align:right'>" + flag(f['away']) + " " + f['away'] + "</div>"
        fix += "</div>"
        fix += "<div style='display:flex;height:8px;border-radius:5px;overflow:hidden'>"
        fix += "<div style='background:#7c3aed;flex:" + str(f['hp']) + "'></div>"
        fix += "<div style='background:#ddd;flex:" + str(f['dp']) + "'></div>"
        fix += "<div style='background:#ec4899;flex:" + str(f['ap']) + "'></div>"
        fix += "</div>"
        fix += "<div style='display:flex;justify-content:space-between;font-size:0.8em;color:#888;margin-top:4px'>"
        fix += "<span>" + str(f['hp']) + "%</span>"
        fix += "<span>" + str(f['dp']) + "%</span>"
        fix += "<span>" + str(f['ap']) + "%</span>"
        fix += "</div></div>"

    # Groups
    grps = "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:15px'>"
    for g, ts in GROUPS.items():
        grps += "<div style='background:white;border-radius:12px;padding:15px;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        grps += "<h3 style='color:#7c3aed;margin-bottom:10px;font-size:1em'>Group " + g + "</h3>"
        for t in ts:
            grps += "<div style='padding:8px;border-bottom:1px solid #f0f0f0;display:flex;align-items:center;gap:8px'>"
            grps += "<span style='font-size:1.3em'>" + flag(t) + "</span>"
            grps += "<span style='font-size:0.9em'>" + t + "</span></div>"
        grps += "</div>"
    grps += "</div>"

    # Bracket
    bracket = "<div style='overflow-x:auto'>"
    bracket += "<div style='min-width:700px;padding:20px'>"
    bracket += "<h3 style='color:#7c3aed;margin-bottom:20px;text-align:center'>Round of 16</h3>"
    bracket += "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px'>"
    for grp, ts in GROUPS.items():
        bracket += "<div style='background:white;border-radius:10px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        bracket += "<div style='font-size:0.7em;color:#7c3aed;font-weight:bold;margin-bottom:8px'>Group " + grp + "</div>"
        for t in ts:
            bracket += "<div style='display:flex;align-items:center;gap:6px;padding:4px 0;border-bottom:1px solid #f0f0f0'>"
            bracket += "<span>" + flag(t) + "</span>"
            bracket += "<span style='font-size:0.8em'>" + t + "</span></div>"
        bracket += "</div>"
    bracket += "</div>"

    bracket += "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px'>"
    for i in range(4):
        bracket += "<div style='background:#f3e8ff;border-radius:10px;padding:15px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        bracket += "<div style='color:#7c3aed;font-weight:bold'>Last 16</div>"
        bracket += "<div style='color:#888;font-size:0.8em;margin-top:5px'>TBD</div></div>"
    bracket += "</div>"

    bracket += "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:20px'>"
    for label in ["Quarter Final", "Quarter Final"]:
        bracket += "<div style='background:#ede9fe;border-radius:10px;padding:15px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        bracket += "<div style='color:#7c3aed;font-weight:bold'>" + label + "</div>"
        bracket += "<div style='color:#888;font-size:0.8em;margin-top:5px'>TBD</div></div>"
    bracket += "</div>"

    bracket += "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:20px'>"
    for label in ["Semi Final", "Semi Final"]:
        bracket += "<div style='background:#ddd6fe;border-radius:10px;padding:15px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>"
        bracket += "<div style='color:#7c3aed;font-weight:bold'>" + label + "</div>"
        bracket += "<div style='color:#888;font-size:0.8em;margin-top:5px'>TBD</div></div>"
    bracket += "</div>"

    bracket += "<div style='background:linear-gradient(135deg,#7c3aed,#ec4899);border-radius:10px;padding:20px;text-align:center;color:white;margin-bottom:10px'>"
    bracket += "<div style='font-weight:bold;font-size:1.2em'>🏆 Final</div>"
    bracket += "<div style='font-size:0.9em;margin-top:5px;opacity:0.8'>Winner: " + flag(champ[0]['team']) + " " + champ[0]['team'] + "?</div></div>"

    bracket += "<div style='background:#fce7f3;border-radius:10px;padding:15px;text-align:center'>"
    bracket += "<div style='color:#ec4899;font-weight:bold;font-size:1.1em'>🥇 Predicted Champion</div>"
    bracket += "<div style='font-size:2em;margin:10px 0'>" + flag(champ[0]['team']) + "</div>"
    bracket += "<div style='font-weight:bold;font-size:1.2em'>" + champ[0]['team'] + "</div>"
    bracket += "<div style='color:#888;font-size:0.9em'>" + str(champ[0]['champ']) + "% chance</div></div>"
    bracket += "</div></div>"

    page = "<!DOCTYPE html><html><head><title>World Cup 2026</title>"
    page += "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    page += "<style>"
    page += "*{margin:0;padding:0;box-sizing:border-box;}"
    page += "body{font-family:Arial;background:#f5f5f5;}"
    page += ".hdr{background:linear-gradient(135deg,#1a0533,#7c3aed);padding:25px;text-align:center;color:white;}"
    page += ".hdr h1{font-size:2em;}"
    page += ".tabs{display:flex;background:white;border-bottom:2px solid #eee;position:sticky;top:0;z-index:100;}"
    page += ".tab{flex:1;padding:15px;text-align:center;cursor:pointer;font-weight:bold;color:#888;border-bottom:3px solid transparent;font-size:0.85em;}"
    page += ".tab.active{color:#7c3aed;border-bottom:3px solid #7c3aed;}"
    page += ".cnt{display:none;padding:20px;max-width:900px;margin:0 auto;}"
    page += ".cnt.active{display:block;}"
    page += "table{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);}"
    page += "th{padding:12px;text-align:left;background:#f9f9f9;color:#888;font-size:0.85em;}"
    page += "td{padding:12px;border-top:1px solid #f0f0f0;}"
    page += "h2{margin:20px 0;color:#7c3aed;}"
    page += "</style></head><body>"
    page += "<div class='hdr'><h1>⚽ World Cup 2026</h1>"
    page += "<p style='color:#ddd;margin-top:5px'>AI Powered Predictions</p></div>"
    page += "<div class='tabs'>"
    page += "<div class='tab active' onclick=\"show('s',this)\">🏆 Standings</div>"
    page += "<div class='tab' onclick=\"show('f',this)\">📅 Fixtures</div>"
    page += "<div class='tab' onclick=\"show('g',this)\">👥 Groups</div>"
    page += "<div class='tab' onclick=\"show('b',this)\">🎯 Bracket</div>"
    page += "</div>"
    page += "<div id='s' class='cnt active'><h2>Championship Predictions</h2>"
    page += "<table><tr><th>#</th><th>Team</th><th>Final</th><th>Champion</th></tr>"
    page += rows + "</table></div>"
    page += "<div id='f' class='cnt'><h2>Group Fixtures</h2>" + fix + "</div>"
    page += "<div id='g' class='cnt'><h2>Groups</h2>" + grps + "</div>"
    page += "<div id='b' class='cnt'><h2>Tournament Bracket</h2>" + bracket + "</div>"
    page += "<script>function show(id,el){document.querySelectorAll('.cnt').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(id).classList.add('active');el.classList.add('active');}</script>"
    page += "</body></html>"
    return page

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    