import re

with open('model.py', 'r') as f:
    content = f.read()

# Update team_stats initialization
content = content.replace(
    "hs = team_stats.setdefault(home, {'history': [], 'gd': []})\n    as_ = team_stats.setdefault(away, {'history': [], 'gd': []})",
    "hs = team_stats.setdefault(home, {'history': [], 'comp_history': [], 'gd': []})\n    as_ = team_stats.setdefault(away, {'history': [], 'comp_history': [], 'gd': []})"
)

# Update the feature calculation loop
old_form_code = """    # Form: win rate in last 10 games
    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    a_form = sum([1 for r in as_['history'][-10:] if r == 'W']) / 10.0 if as_['history'] else 0.45"""

new_form_code = """    # Form: win rate in last 10 games
    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    a_form = sum([1 for r in as_['history'][-10:] if r == 'W']) / 10.0 if as_['history'] else 0.45
    
    # Comp Form: win rate in last 10 competitive games
    h_comp = sum([1 for r in hs['comp_history'][-10:] if r == 'W']) / 10.0 if hs['comp_history'] else 0.45
    a_comp = sum([1 for r in as_['comp_history'][-10:] if r == 'W']) / 10.0 if as_['comp_history'] else 0.45
    
    # Overall Win Rate
    h_wr = sum([1 for r in hs['history'] if r == 'W']) / len(hs['history']) if hs['history'] else 0.45
    a_wr = sum([1 for r in as_['history'] if r == 'W']) / len(as_['history']) if as_['history'] else 0.45"""
content = content.replace(old_form_code, new_form_code)

# Append to lists
old_append = """    feats['h_form'].append(h_form)
    feats['a_form'].append(a_form)"""
new_append = """    feats.setdefault('h_form', []).append(h_form)
    feats.setdefault('a_form', []).append(a_form)
    feats.setdefault('h_comp', []).append(h_comp)
    feats.setdefault('a_comp', []).append(a_comp)
    feats.setdefault('h_wr', []).append(h_wr)
    feats.setdefault('a_wr', []).append(a_wr)"""
content = content.replace(old_append, new_append)

# Update state history to include comp_history
old_state = """    if row['result'] == 'Home Win':
        hs['history'].append('W'); as_['history'].append('L'); hist.append(home)
    elif row['result'] == 'Away Win':
        hs['history'].append('L'); as_['history'].append('W'); hist.append(away)
    else:
        hs['history'].append('D'); as_['history'].append('D'); hist.append('Draw')"""
new_state = """    res_h = 'W' if row['result'] == 0 else ('L' if row['result'] == 1 else 'D')
    res_a = 'L' if row['result'] == 0 else ('W' if row['result'] == 1 else 'D')
    win_team = home if row['result'] == 0 else (away if row['result'] == 1 else 'Draw')
    
    hs['history'].append(res_h); as_['history'].append(res_a); hist.append(win_team)
    if row['tournament'] != 'Friendly':
        hs['comp_history'].append(res_h)
        as_['comp_history'].append(res_a)"""
content = content.replace(old_state, new_state)
# Fix the previous error if it still exists (result was 'Home Win' instead of 0)
content = content.replace("if row['result'] == 'Home Win':", "") # cleanup if it failed

# Update df column assignments
old_cols = """df_results['h_form'] = feats['h_form']
df_results['a_form'] = feats['a_form']"""
new_cols = """df_results['h_form'] = feats['h_form']
df_results['a_form'] = feats['a_form']
df_results['h_comp'] = feats['h_comp']
df_results['a_comp'] = feats['a_comp']
df_results['h_wr'] = feats['h_wr']
df_results['a_wr'] = feats['a_wr']"""
content = content.replace(old_cols, new_cols)

# Replace the duplicate lines
old_dupes = """df_ml['h_comp'] = df_ml['h_form']
df_ml['a_comp'] = df_ml['a_form']
df_ml['comp_diff'] = df_ml['form_diff']

df_ml['h_form2'] = df_ml['h_form']
df_ml['a_form2'] = df_ml['a_form']
df_ml['form_diff2'] = df_ml['form_diff']"""

new_dupes = """df_ml['comp_diff'] = df_ml['h_comp'] - df_ml['a_comp']
df_ml['h_form2'] = df_ml['h_wr']
df_ml['a_form2'] = df_ml['a_wr']
df_ml['form_diff2'] = df_ml['h_form2'] - df_ml['a_form2']"""
content = content.replace(old_dupes, new_dupes)

# Update latest_stats mapping
old_latest = """    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    h_gd = np.mean(hs['gd'][-10:]) if hs['gd'] else 0.0
    latest_stats[team] = {
        'elo': elo_dict.get(team, 1500),
        'win_rate': h_form,
        'recent_form': h_form,
        'comp_recent': h_form,"""
        
new_latest = """    h_form = sum([1 for r in hs['history'][-10:] if r == 'W']) / 10.0 if hs['history'] else 0.45
    h_comp = sum([1 for r in hs['comp_history'][-10:] if r == 'W']) / 10.0 if hs['comp_history'] else 0.45
    h_wr = sum([1 for r in hs['history'] if r == 'W']) / len(hs['history']) if hs['history'] else 0.45
    h_gd = np.mean(hs['gd'][-10:]) if hs['gd'] else 0.0
    latest_stats[team] = {
        'elo': elo_dict.get(team, 1500),
        'win_rate': h_wr,
        'recent_form': h_form,
        'comp_recent': h_comp,"""
content = content.replace(old_latest, new_latest)

with open('model.py', 'w') as f:
    f.write(content)

