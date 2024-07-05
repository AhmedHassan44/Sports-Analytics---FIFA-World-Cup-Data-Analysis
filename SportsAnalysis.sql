-- Step 1: Connect to the sports_analytics database
USE sports_analytics;

------------------------------------------------------------------------------------------------------------------
-- Query 1: Counting Matches Played and Goals Scored
SELECT 
    `Player Name`, 
    COUNT(*) AS NumMatchesPlayed,                    -- Count total matches played by each player
    SUM(Event LIKE '%G%') AS NumGoalsScored          -- Sum of goals scored (Event containing 'G')
FROM 
    WorldCupPlayers
GROUP BY 
    `Player Name`
ORDER BY 
    NumGoalsScored DESC;                             -- Order by goals scored in descending order

------------------------------------------------------------------------------------------------------------------
-- Query 2: Counting Matches Played and Goals Scored by Each Team
SELECT 
    `Home Team Name` AS TeamName,
    COUNT(*) AS NumMatchesPlayed,                    -- Count total matches played by each team as home team
    SUM(`Home Team Goals`) AS GoalsScoredHome,       -- Sum of goals scored by the home team
    SUM(`Away Team Goals`) AS GoalsScoredAway         -- Sum of goals scored by the away team
FROM 
    WorldCupMatches
GROUP BY 
    `Home Team Name`
UNION
SELECT 
    `Away Team Name` AS TeamName,
    COUNT(*) AS NumMatchesPlayed,                    -- Count total matches played by each team as away team
    SUM(`Away Team Goals`) AS GoalsScoredHome,       -- Sum of goals scored by the away team
    SUM(`Home Team Goals`) AS GoalsScoredAway         -- Sum of goals scored by the home team
FROM 
    WorldCupMatches
GROUP BY 
    `Away Team Name`
ORDER BY 
    NumMatchesPlayed DESC;                            -- Order by total matches played in descending order

------------------------------------------------------------------------------------------------------------------
-- Query 3: Total Goals Scored in Each World Cup Edition
SELECT 
    `Year`, 
    SUM(`GoalsScored`) AS TotalGoalsScored           -- Sum of goals scored in each World Cup
FROM 
    WorldCups
GROUP BY 
    `Year`
ORDER BY 
    `Year` DESC;                                     -- Order by World Cup year in descending order

------------------------------------------------------------------------------------------------------------------
-- Query: Average Goals per Match for Top 10 Scorers (Alternative Approach)
SELECT 
    wc.`Player Name`, 
    COUNT(*) AS NumMatchesPlayed,                    -- Count total matches played by each player
    SUM(wc.Event LIKE '%G%') AS NumGoalsScored,      -- Sum of goals scored (Event containing 'G')
    ROUND(SUM(wc.Event LIKE '%G%') / COUNT(*), 2) AS AvgGoalsPerMatch   -- Average goals per match rounded to 2 decimal places
FROM 
    WorldCupPlayers wc
JOIN (
    SELECT 
        `Player Name`,
        SUM(Event LIKE '%G%') AS TotalGoalsScored
    FROM 
        WorldCupPlayers
    GROUP BY 
        `Player Name`
    ORDER BY 
        TotalGoalsScored DESC
    LIMIT 10  -- Select top 10 players by total goals scored
) top_players ON wc.`Player Name` = top_players.`Player Name`
GROUP BY 
    wc.`Player Name`
ORDER BY 
    AvgGoalsPerMatch DESC;                             -- Order by average goals per match in descending order

