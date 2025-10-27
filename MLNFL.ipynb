import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("yearly_player_stats_defense.csv")
df = df[df['player_name'] != 'N/A']

POS = "CB"  # change to "LB", "CB", "DT"

data = df

dropped = ['team', 'player_id', 'player_name', 'college', 'conference', 'division',
           'career_fantasy_points_ppr', 'career_fantasy_points_standard',
           'season_average_fantasy_points_ppr', 'career_average_fantasy_points_ppr',
           'season_average_fantasy_points_standard', 'career_average_fantasy_points_standard',
           ]

data = data.drop(columns=dropped)
data['games_played_per_season'] = data['games_played_career'] / data['seasons_played']
data = data[data['games_played_per_season'] >= 10]
data = data.drop(columns=['games_played_per_season'])
data = data[data['games_missed'] <= 7]

delta_cols = [
    'delta_def_touchdown',
    'delta_fantasy_points_ppr',
    'delta_fantasy_points_standard',
    'delta_defense_snaps',
    'delta_team_defense_snaps',
    'delta_depth_team',
    'delta_games_missed',
    'delta_defense_pct',
    'delta_career_solo_tackle',
    'delta_career_assist_tackle',
    'delta_career_tackle_with_assist',
    'delta_career_sack',
    'delta_career_qb_hit',
    'delta_career_safety',
    'delta_career_interception',
    'delta_career_def_touchdown',
    'delta_career_defensive_two_point_attempt',
    'delta_career_fumble_forced',
    'delta_career_defensive_two_point_conv',
    'delta_career_defensive_extra_point_attempt',
    'delta_career_defensive_extra_point_conv',
    'delta_career_fantasy_points_ppr',
    'delta_career_fantasy_points_standard',
    'delta_career_defense_snaps',
    'delta_career_team_defense_snaps',
    'delta_season_average_solo_tackle',
    'delta_career_average_solo_tackle',
    'delta_season_average_assist_tackle',
    'delta_career_average_assist_tackle',
    'delta_season_average_tackle_with_assist',
    'delta_career_average_tackle_with_assist',
    'delta_season_average_sack',
    'delta_career_average_sack',
    'delta_season_average_qb_hit',
    'delta_career_average_qb_hit',
    'delta_season_average_safety',
    'delta_career_average_safety',
    'delta_season_average_interception',
    'delta_career_average_interception',
    'delta_season_average_def_touchdown',
    'delta_career_average_def_touchdown',
    'delta_season_average_defensive_two_point_attempt',
    'delta_career_average_defensive_two_point_attempt',
    'delta_season_average_fumble_forced',
    'delta_career_average_fumble_forced',
    'delta_season_average_defensive_two_point_conv',
    'delta_career_average_defensive_two_point_conv',
    'delta_season_average_defensive_extra_point_attempt',
    'delta_career_average_defensive_extra_point_attempt',
    'delta_season_average_defensive_extra_point_conv',
    'delta_career_average_defensive_extra_point_conv',
    'delta_season_average_fantasy_points_ppr',
    'delta_career_average_fantasy_points_ppr',
    'delta_season_average_fantasy_points_standard',
    'delta_career_average_fantasy_points_standard',
    'delta_season_average_defense_snaps',
    'delta_career_average_defense_snaps',
    'delta_season_average_team_defense_snaps',
    'delta_career_average_team_defense_snaps'
]

season_average_features = [
    'season_average_solo_tackle',
    'season_average_assist_tackle',
    'season_average_tackle_with_assist',
    'season_average_sack',
    'season_average_qb_hit',
    'season_average_safety',
    'season_average_interception',
    'season_average_def_touchdown',
    'season_average_defensive_two_point_attempt',
    'season_average_fumble_forced',
    'season_average_defensive_two_point_conv',
    'season_average_defensive_extra_point_attempt',
    'season_average_defensive_extra_point_conv',
    'season_average_defense_snaps',
    'season_average_team_defense_snaps'
]


"""
data['fantasy_points_ppr'] = (
    data['delta_career_solo_tackle'] * 0.5 +
    data['delta_career_assist_tackle'] * 0.15 +
    data['delta_career_sack'] * 1.0 +
    data['delta_career_interception'] * 4.0 +
    data['delta_career_fumble_forced'] * 3.0 +
    data['delta_career_def_touchdown'] * 6.0 +
    data['delta_career_safety'] * 2.0
)"""

# --- Opportunity scaling (unchanged logic) ---
snap_ratio = np.minimum(
    (data['defense_snaps'] / data['defense_snaps'].replace(0, np.nan).mean()).fillna(0),
    1.0
)
pct_ratio = data['defense_pct'].clip(lower=0, upper=1).fillna(0)
opp_raw = 0.6 * snap_ratio + 0.4 * pct_ratio
opp = (0.75 + 0.5 * opp_raw).clip(lower=0.75, upper=1.25)

data['fantasy_points_ppr'] = (
    data['season_average_solo_tackle'] * 0.30 +
    data['season_average_assist_tackle'] * 0.10 +
    data['season_average_sack'] * 1.50 +
    data['season_average_qb_hit'] * 0.50 +
    data['season_average_interception'] * 5.50 +
    data['season_average_fumble_forced'] * 3.00 +
    data['season_average_def_touchdown'] * 6.00 +
    data['season_average_safety'] * 2.00
) * opp

data = data.drop(columns=delta_cols)
data = data.drop(columns=season_average_features)
# Define features (X) and target (y)
pos_data = data[data['position'] == POS]

y = pos_data['fantasy_points_ppr'] # target variable
X = pos_data.drop(columns=['fantasy_points_ppr', 'fantasy_points_standard', 'position'])  # predictors

# chronological train/test split by season
max_season = X['season'].max()
train_mask = X['season'] < max_season
test_mask  = X['season'] == max_season
# Save player + season info before dropping
X_test_with_season = X.loc[test_mask][['season']].copy()

X_train = X.loc[train_mask].drop(columns=['season'], errors='ignore')
X_test  = X.loc[test_mask].drop(columns=['season'], errors='ignore')
y_train = y.loc[train_mask]
y_test  = y.loc[test_mask]

print(f"Training on seasons < {max_season}, testing on season {max_season}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# Add predictions and actuals back into your test set
X_test_copy = X_test.copy()
X_test_copy['Actual'] = y_test.values
X_test_copy['Predicted'] = y_pred

player_info = df[['player_id', 'player_name', 'season']]
pos_info = player_info.loc[pos_data.index]

test_indices = X.index[test_mask]  # indices of the test fold
pos_info_test = pos_info.loc[test_indices]

X_test_copy = X_test_copy.join(pos_info_test[['player_name', 'season']])


# Choose which season to display
target_season = X['season'].max()

# Filter and sort by predicted fantasy points
top_players = X_test_copy.sort_values(by='Predicted', ascending=False).head(15)
print(f"\n Top 15 Predicted Defensive Players for {max_season}:")
print(top_players[['player_name', 'Predicted', 'Actual']].to_string(index=False))

# Compute average predicted POSITION fantasy points per team for the test season
team_info = df[['player_id', 'team', 'season']]
pos_team_info = team_info.loc[pos_data.index]
pos_team_test = pos_team_info.loc[test_indices]

X_test_copy = X_test_copy.join(pos_team_test['team'])

team_avg = (
    X_test_copy.groupby('team')[['Predicted', 'Actual']]
    .mean()
    .sort_values(by='Predicted', ascending=False)
)

print(f"\n Average Predicted {POS} Fantasy Points per Team for {int(max_season)}:")
print(team_avg.head(15).to_string())

team_avg.to_csv(f"avg_{POS}_fantasy_by_team_{int(max_season)}.csv")
print(f"\nSaved as avg_{POS}_fantasy_by_team_{int(max_season)}.csv")

# After building X_test_copy with player_name + Predicted:
tiers = pd.read_csv("cornerback_tiers_2024.csv")  # player_name, team, Tier, TierScore
eval_df = X_test_copy.merge(tiers[['player_name','Tier','TierScore']], on='player_name', how='inner')

from scipy.stats import spearmanr
rho, p = spearmanr(eval_df['Predicted'], eval_df['TierScore'])
print(f"Spearman rank corr (Predicted vs TierScore): rho={rho:.3f}, p={p:.3g}")

