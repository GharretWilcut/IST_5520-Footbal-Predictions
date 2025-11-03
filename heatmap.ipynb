import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

offenseDF = pd.read_csv("yearly_team_stats_offense.csv")
defenseDF = pd.read_csv("yearly_team_stats_defense.csv")
offenseDF = offenseDF[offenseDF['season_type'] != 'POST']
defenseDF = defenseDF[defenseDF['season_type'] != 'POST']
df = pd.merge(offenseDF, defenseDF, on=['team','season', 'season_type', 'win_pct'], suffixes=('_off','_def'))


target = 'win_pct'
offense_data = offenseDF.drop(columns='win_pct')
defense_data = defenseDF.drop(columns='win_pct')

team_data = pd.merge(offense_data, defense_data, on=['team','season', 'season_type'], suffixes=('_off','_def'))
offense_avg_cols = [c for c in team_data.columns if c.startswith('average_')]
defense_avg_cols = [c for c in team_data.columns if c.startswith('average_')]

drop = ['team', 'season_type', 'win_off', 'win_def', 'loss_off',
        'loss_def', 'tie_off', 'tie_def', 'record_off', 'record_def',
        'win_pct_off', 'win_pct_def']
team_data = team_data.drop(columns=drop, errors='ignore')

# Define features (X) and target (y)
y_col = df[target]
X_cols = team_data

latest_season = int(df["season"].max())

# ---------- Build matchup training pairs from past seasons ----------
pairs_X, pairs_y = [], []
for season, G in df[df["season"] != latest_season].groupby("season"):
    G = G.reset_index(drop=True)
    Xs = G[X_cols.columns].to_numpy()
    ys = G[y_col.name].to_numpy().astype(float)
    for i, j in combinations(range(len(G)), 2):
        d = Xs[i] - Xs[j]
        y = 1 if ys[i] > ys[j] else 0
        pairs_X.append(d); pairs_y.append(y)
        pairs_X.append(-d); pairs_y.append(1 - y)

pairs_X = np.asarray(pairs_X)
pairs_y = np.asarray(pairs_y)

# ---------- Train Naive Bayes on standardized differences ----------
nb = make_pipeline(StandardScaler(with_mean=True, with_std=True), GaussianNB())
nb.fit(pairs_X, pairs_y)

# ---------- Choose latest-season Top-14 teams ----------
cur = df[df["season"] == latest_season].set_index("team")
top14 = (cur[y_col.name].sort_values(ascending=False).head(14)).index.tolist()
X_cur = cur.loc[top14, X_cols.columns].to_numpy()

# ---------- Build win-probability matrix P(row beats col) ----------
n = len(top14)
prob = np.zeros((n, n))
for a in range(n):
    for b in range(n):
        if a == b:
            prob[a, b] = 0.5
        else:
            prob[a, b] = nb.predict_proba((X_cur[a] - X_cur[b]).reshape(1, -1))[0, 1]

# ---------- Plot heat map ----------
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(prob, aspect='auto')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(top14, rotation=45, ha='right')
ax.set_yticklabels(top14)
ax.set_title(f"Naive Bayes Win Probability Heatmap — {latest_season} (Row beats Column)")
cbar = plt.colorbar(im, ax=ax); cbar.set_label("P(Row team beats Column)")
plt.tight_layout()
plt.show()
