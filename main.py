# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# Connect to MySQL and load the WorldCupMatches data
engine = create_engine('mysql+mysqlconnector://root:17#Mysql17@localhost/sports_analytics')
matches_df = pd.read_sql_table('WorldCupMatches', con=engine)

# Data Preparation: Aggregate team performance metrics
team_performance = pd.DataFrame()

# Analysis to calculate team metrics
# Calculate total goals scored by each team
home_goals = matches_df.groupby('Home Team Name')['Home Team Goals'].sum()
away_goals = matches_df.groupby('Away Team Name')['Away Team Goals'].sum()
total_goals = home_goals.add(away_goals, fill_value=0)
team_performance['Total Goals'] = total_goals

# Calculate wins and losses for each team
home_wins = matches_df[matches_df['Home Team Goals'] > matches_df['Away Team Goals']].groupby('Home Team Name').size()
away_wins = matches_df[matches_df['Away Team Goals'] > matches_df['Home Team Goals']].groupby('Away Team Name').size()
total_wins = home_wins.add(away_wins, fill_value=0)
team_performance['Wins'] = total_wins

home_losses = matches_df[matches_df['Home Team Goals'] < matches_df['Away Team Goals']].groupby('Home Team Name').size()
away_losses = matches_df[matches_df['Away Team Goals'] < matches_df['Home Team Goals']].groupby('Away Team Name').size()
total_losses = home_losses.add(away_losses, fill_value=0)
team_performance['Losses'] = total_losses

# Calculate average attendance per match for each team
attendance_per_match = matches_df.groupby('Home Team Name')['Attendance'].mean()
team_performance['Avg Attendance'] = attendance_per_match

# Handling missing values and scaling features
team_performance.fillna(0, inplace=True)  # Assuming no attendance data means no matches played
scaler = StandardScaler()
scaled_features = scaler.fit_transform(team_performance)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to team_performance dataframe
team_performance['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Wins', y='Total Goals', hue='Cluster', data=team_performance, palette='viridis', s=100)
plt.title('Clustering of Teams Based on Performance')
plt.xlabel('Total Wins')
plt.ylabel('Total Goals Scored')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Aggregate team statistics
home_team_stats = matches_df.groupby('Home Team Name').agg({
    'Home Team Goals': 'sum',
    'Away Team Goals': 'sum',
    'MatchID': 'count'
}).reset_index()

home_team_stats.columns = ['Team', 'Goals Scored', 'Goals Conceded', 'Matches Played']

away_team_stats = matches_df.groupby('Away Team Name').agg({
    'Home Team Goals': 'sum',
    'Away Team Goals': 'sum',
    'MatchID': 'count'
}).reset_index()

away_team_stats.columns = ['Team', 'Goals Conceded', 'Goals Scored', 'Matches Played']

team_stats = pd.concat([home_team_stats, away_team_stats], axis=0, ignore_index=True, sort=False)
team_stats = team_stats.groupby('Team').agg({
    'Goals Scored': 'sum',
    'Goals Conceded': 'sum',
    'Matches Played': 'sum'
}).reset_index()

# Plotting goals scored vs matches played
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Matches Played', y='Goals Scored', data=team_stats, s=100, color='blue', alpha=0.7)
plt.title('Goals Scored vs Matches Played by Each Team in World Cup')
plt.xlabel('Matches Played')
plt.ylabel('Goals Scored')
plt.grid(True)
plt.tight_layout()
plt.show()

# Attendance by year
attendance_by_year = matches_df.groupby('Year')['Attendance'].sum().reset_index()

# Plotting attendance by year
plt.figure(figsize=(12, 8))
sns.barplot(x='Year', y='Attendance', data=attendance_by_year, palette='viridis')
plt.title('Total Attendance at World Cup Matches by Year')
plt.xlabel('Year')
plt.ylabel('Total Attendance')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Load and process WorldCupPlayers data
world_cup_players = pd.read_csv('WorldCupPlayers.csv')

# Filter rows where 'Event' column contains 'G'
goals_scored = world_cup_players[world_cup_players['Event'].str.contains('G', na=False)]
num_goals_scored = goals_scored.groupby('Player Name')['Event'].count().reset_index()
num_goals_scored.columns = ['Player Name', 'NumGoalsScored']
num_goals_scored = num_goals_scored.sort_values(by='NumGoalsScored', ascending=False)

# Take top 10 scorers for visualization
top_scorers = num_goals_scored.head(10)

# Plotting top goal scorers with Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='NumGoalsScored', y='Player Name', data=top_scorers, palette='viridis')
plt.title('Top 10 Goal Scorers in World Cup')
plt.xlabel('Number of Goals Scored')
plt.ylabel('Player Name')
plt.show()

# Hypothesis testing: Group stage vs Knockout stage goals
group_stage_goals = matches_df[matches_df['Stage'].str.contains('Group')]
knockout_stage_goals = matches_df[~matches_df['Stage'].str.contains('Group')]

t_stat, p_value = ttest_ind(group_stage_goals['Home Team Goals'] + group_stage_goals['Away Team Goals'],
                            knockout_stage_goals['Home Team Goals'] + knockout_stage_goals['Away Team Goals'])

print(f"T-test results:")
print(f"  - T-statistic: {t_stat:.4f}")
print(f"  - P-value: {p_value:.4f}")

alpha = 0.05  # Significance level
if p_value < alpha:
    print("Conclusion: Reject null hypothesis. There is a significant difference in average goals scored between group stage and knockout stage matches.")
else:
    print("Conclusion: Fail to reject null hypothesis. There is no significant difference in average goals scored between group stage and knockout stage matches.")

# Aggregate goals scored by teams
matches_df = matches_df[['Year', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals']]
goals_by_team = {}

for idx, row in matches_df.iterrows():
    home_team = row['Home Team Name']
    away_team = row['Away Team Name']
    home_goals = row['Home Team Goals']
    away_goals = row['Away Team Goals']

    if home_team not in goals_by_team:
        goals_by_team[home_team] = 0
    if away_team not in goals_by_team:
        goals_by_team[away_team] = 0

    goals_by_team[home_team] += home_goals
    goals_by_team[away_team] += away_goals

goals_df = pd.DataFrame(list(goals_by_team.items()), columns=['Team', 'Total Goals'])
goals_df = goals_df.sort_values(by='Total Goals', ascending=False).head(10)

# Visualize top 10 most goal-scoring teams
plt.figure(figsize=(12, 8))
sns.barplot(x='Total Goals', y='Team', data=goals_df, palette='viridis')
plt.title('Top 10 Most Goal-Scoring Teams in World Cup History')
plt.xlabel('Total Goals')
plt.ylabel('Team')
plt.grid(True)
plt.tight_layout()
plt.show()

# Top 10 players with most matches played
players_matches = matches_df['Player Name'].value_counts().reset_index()
players_matches.columns = ['Player Name', 'Matches Played']
top_10_players_matches = players_matches.head(10)

# Visualize top 10 players by matches played
plt.figure(figsize=(10, 6))
sns.barplot(x='Matches Played', y='Player Name', data=top_10_players_matches, palette='viridis')
plt.title('Top 10 Players by Matches Played in World Cup')
plt.xlabel('Matches Played')
plt.ylabel('Player Name')
plt.show()

# Top 10 teams with most matches played
teams_matches = matches_df['Home Team Name'].append(matches_df['Away Team Name']).value_counts().reset_index()
teams_matches.columns = ['Team', 'Matches Played']
top_10_teams_matches = teams_matches.head(10)

# Visualize top 10 teams by matches played
plt.figure(figsize=(10, 6))
sns.barplot(x='Matches Played', y='Team', data=top_10_teams_matches, palette='viridis')
plt.title('Top 10 Teams by Matches Played in World Cup')
plt.xlabel('Matches Played')
plt.ylabel('Team')
plt.show()
