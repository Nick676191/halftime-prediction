from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import questionary
from sklearn.model_selection import train_test_split

# Have to load the environment variable prior to loading the kagglehub and kaggle libraries in python
load_dotenv()
import kagglehub
import kaggle
kaggle.api.authenticate()

def kaggleDfLoader(acctStr: str, dfStr: str):
    # Load a DataFrame with a specific version of a CSV
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        acctStr,
        dfStr
    )

    return df


def dfDescriber(df) -> None:
    print(df.head())
    print("-"*60)
    print()

    shape = df.shape
    print(f"This dataframe has {shape[0]} rows and {shape[1]} columns.")
    print("-"*60)
    print()

    print("Each of the columns have this many null values")
    print(df.isna().sum())
    print("-"*60)
    print()

    print(df.dtypes)
    print("-"*60)
    print()

    for dtype in df.dtypes.drop_duplicates():
        print(f"There are {sum([1 if dt==dtype else 0 for dt in df.dtypes])} columns of the {dtype} data-type")
    print("-"*60)
    print()

    # # looking into the string columns
    # str_df = df.loc[:, df.dtypes == "str"]
    # print(str_df.head(15))
    # results_dict = {}
    # for col in str_df.columns[3:14]:
    #     print(f"The unique values in {col} are {str_df[col].unique()}")
    #     results_dict[col] = str_df[col].unique()
    #     ind_dict = {}
    #     for val in str_df[col].unique():
    #         number = sum([1 if x==val else 0 for x in str_df[col]])
    #         ind_dict[val] = number
    #         print(f"{val} shows up {number} times.")
    #     results_dict[col] = ind_dict

    # print(results_dict)
    # print("-"*60)
    # print()


def dfVisualizer(df) -> None:
    print()


def dfChecker(df) -> None:
    # code below proves that there is a column that tracks goals scored and a column for goals conceded
    # this column doesn't include the goals scored and conceded during the last match of the season for each team 
    check_df = df[(df["Date"].str[-2:] == "00") | ((df["Date"].str[-2:] == "01") & (df["MW"] > 20))]
    print(check_df.tail(15))
    print(check_df["MW"].tail(15))

    check_df = check_df[(check_df["HomeTeam"] == "Charlton") | (check_df["AwayTeam"] == "Charlton")]
    print(check_df.head())

    charlton_hgs = check_df.loc[check_df["HomeTeam"] == "Charlton", "FTHG"]
    charlton_ags = check_df.loc[check_df["AwayTeam"] == "Charlton", "FTAG"]
    charlton_hgc = check_df.loc[check_df["HomeTeam"] != "Charlton", "FTHG"]
    charlton_agc = check_df.loc[check_df["AwayTeam"] != "Charlton", "FTAG"]

    print(f"Charlton scored {sum(charlton_hgs) + sum(charlton_ags)} goals during the 00/01 season.")
    print(f"Charlton conceded {sum(charlton_hgc) + sum(charlton_agc)} goals during the 00/01 season.")

    print(check_df.iloc[-1, :])

    if check_df.iloc[-1, 2] == "Charlton":
        assert(check_df.iloc[-1, 7] == (sum(charlton_hgs) + sum(charlton_ags)) - check_df.iloc[-1, 4])
        assert(check_df.iloc[-1, 9] == (sum(charlton_hgc) + sum(charlton_agc)) - check_df.iloc[-1, 5])

    if check_df.iloc[-1, 3] == "Charlton":
        assert(check_df.iloc[-1, 8] == (sum(charlton_hgs) + sum(charlton_ags)) - check_df.iloc[-1, 5])
        assert(check_df.iloc[-1, 10] == (sum(charlton_hgc) + sum(charlton_agc)) - check_df.iloc[-1, 4])


def initial_preprocessDf(df):
    df = df.drop(["Unnamed: 0", "HTFormPtsStr", "ATFormPtsStr"], axis=1)

    # change results column to include ties, instead of all draws being assigned to NH
    df.loc[df["FTHG"] == df["FTAG"], "FTR"] = "D"

    # rename columns so that they make more sense
    df = df.rename(columns={
        "FTHG": "full_time_match_ht_goals",
        "FTAG": "full_time_match_at_goals",
        "FTR": "ht_result",
        "HTGS": "ht_goals_scored",
        "ATGS": "at_goals_scored",
        "HTGC": "ht_goals_conceded",
        "ATGC": "at_goals_conceded",
        "HTP": "ht_avg_ppg_played",
        "ATP": "at_avg_ppg_played",
        "MW": "match_week",

        })

    return df


def feature_engineer(df):
    season_dict = {}
    # create a feature that tracks the points in the respective season for each team and each team's respective ranking
    # this dataset goes until the year of 2018, starting in 2000. Each season starts in august and ends around may with 38 match weeks
    for current_year in range(18):
        if len(str(current_year)) == 1:
            current_year = str(0) + str(current_year)
        else:
            current_year = str(current_year)
        if current_year[-1] == "9":
            next_year_first_val = int(current_year[0]) + 1
            next_year_last_val = 0
            next_year = str(next_year_first_val) + str(next_year_last_val)
        else:
            next_year_last_val = int(current_year[-1]) + 1
            next_year = str(current_year[0]) + str(next_year_last_val)
        print(f"FEATURE ENGINEERING: Working with {current_year}/{next_year} Season.")

        # seperate out the dataframe for each season from 00/01 to 17/18
        first_year_df = df[df["Date"].str[-2:] == current_year]
        match_week_num = first_year_df.iloc[-1, first_year_df.columns.get_loc("match_week")]
        print(f"The {current_year} part of the season ends at match week {match_week_num}")
        df["temp_date"] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        season_df = df[((df["Date"].str[-2:] == current_year) & (df["temp_date"].dt.month >= 8)) | ((df["Date"].str[-2:] == next_year) & (df["temp_date"].dt.month <= 7))]
        df = df.drop(columns=["temp_date"])
        # checking that each season df is actually ending at the end of the season
        if (season_df["match_week"].tail(1) != 38).all():
            season_df["temp_date"] = pd.to_datetime(season_df['Date'], format='mixed', dayfirst=True)
            exclude_year = season_df["temp_date"].tail(1).dt.year
            season_df = season_df[~((season_df['temp_date'].dt.month == 12) & (season_df['temp_date'].dt.year == exclude_year.values[0]) & (season_df['match_week'] != 38))]
            season_df = season_df.drop(columns=["temp_date"])
        assert(season_df["match_week"].head(1) == 1).all()
        assert(season_df["match_week"].tail(1) == 38).all()
        print("FEATURE ENGINEERING: This season's DataFrame object ends at the correct time")

        # append each season to the dictionary of dfs for processing, will concatenate back together once dfs are processed
        season_dict[f"{current_year}/{next_year} Season"] = season_df

    for season_key in season_dict:
        current_df = season_dict[season_key]
        list_of_teams = current_df["HomeTeam"].unique()
        team_rankings_dict = {team: {"points": 0, "goal_diff": 0, "prev_game": "M"} for team in list_of_teams}

        # update team_rankings dictionary and create rankings columns for home and away teams
        season_ht_points_series = []
        season_at_points_series = []
        ht_ranking = []
        at_ranking = []
        for row in current_df.itertuples():
            # points updates
            if team_rankings_dict[row.HomeTeam]["prev_game"] == "W":
                team_rankings_dict[row.HomeTeam]["points"] = team_rankings_dict[row.HomeTeam]["points"] + 3
            if team_rankings_dict[row.HomeTeam]["prev_game"] == "D":
                team_rankings_dict[row.HomeTeam]["points"] = team_rankings_dict[row.HomeTeam]["points"] + 1
        
            if team_rankings_dict[row.AwayTeam]["prev_game"] == "W":
                team_rankings_dict[row.AwayTeam]["points"] = team_rankings_dict[row.AwayTeam]["points"] + 3
            if team_rankings_dict[row.AwayTeam]["prev_game"] == "D":
                team_rankings_dict[row.AwayTeam]["points"] = team_rankings_dict[row.AwayTeam]["points"] + 1

            # update previous game result for each team
            if row.ht_result == "H":
                team_rankings_dict[row.HomeTeam]["prev_game"] = "W"
                team_rankings_dict[row.AwayTeam]["prev_game"] = "L"
            if row.ht_result == "NH":
                team_rankings_dict[row.HomeTeam]["prev_game"] = "L"
                team_rankings_dict[row.AwayTeam]["prev_game"] = "W"
            if row.ht_result == "D":
                team_rankings_dict[row.HomeTeam]["prev_game"] = "D"
                team_rankings_dict[row.AwayTeam]["prev_game"] = "D"

            # update goal_diff for each team
            team_rankings_dict[row.HomeTeam]["goal_diff"] = row.ht_goals_scored - row.ht_goals_conceded
            team_rankings_dict[row.AwayTeam]["goal_diff"] = row.at_goals_scored - row.at_goals_conceded
            # updating series values for df
            season_ht_points_series.append(team_rankings_dict[row.HomeTeam]["points"])
            season_at_points_series.append(team_rankings_dict[row.AwayTeam]["points"])
            # sorting the rankings of each team
            sorted_team_data = dict(sorted(team_rankings_dict.items(), key=lambda item: (item[1]['points'], item[1]['goal_diff']), reverse=True))
            ht_ranking_num = list(sorted_team_data).index(row.HomeTeam)
            ht_ranking.append(ht_ranking_num + 1)
            at_ranking_num = list(sorted_team_data).index(row.AwayTeam)
            at_ranking.append(at_ranking_num + 1)

        # add series to respective dataframe
        current_df["ht_season_points"] = season_ht_points_series
        current_df["at_season_points"] = season_at_points_series
        current_df["ht_ranking"] = ht_ranking
        current_df["at_ranking"] = at_ranking

        # create a weighted form points column for the away and home teams

        # create a games played column for the home and away teams

        # concatenate all of the dfs back together into one large dataframe again
    
    return season_dict


def mlPreprocessor(df):
    def ordinal_changer(val):
        if val == "M":
            return 0
        elif val == "L":
            return 1
        elif val == "D":
            return 2
        else:
            return 3

    df = df.drop(["Date", "HomeTeam", "AwayTeam"], axis=1)

    cols_to_change = ["HM1", "HM2", "HM3", "HM4", "HM5", "AM1", "AM2", "AM3", "AM4", "AM5"]
    for col in cols_to_change:
        df[col] = df[col].apply(lambda x: ordinal_changer(x))

    return df
    


def data_extract(df):
    y = df["FTR"]
    X = df.drop(["FTR"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4444)

    return (X_train, X_test, y_train, y_test)


def main():
    # # Setting script variables
    # ACCT_STRING = "saife245/english-premier-league"
    # DF_STRING = "final_dataset.csv"

    # Setting new script vars
    ACCT_STRING = "excel4soccer/espn-soccer-data"
    LEAGUES_DF_STRING = "base_data/leagues.csv"
    TEAMS_DF_STRING = "base_data/teams.csv"
    FIXTURES_DF_STRING = "base_data/fixtures.csv"
    STANDINGS_DF_STRING = "base_data/standings.csv"
    TEAM_STATS_PER_FIXTURE = "base_data/teamStats.csv"
    KEY_EVENTS_STRING = "keyEvents_data/keyEvents_2024_ENG.1.csv"
    KEY_EVENT_KEY_DF_STRING = "base_data/keyEventDescription.csv"

    # loading datasets
    # leagues_df = kaggleDfLoader(ACCT_STRING, LEAGUES_DF_STRING)
    teams_df = kaggleDfLoader(ACCT_STRING, TEAMS_DF_STRING)
    fixtures_df = kaggleDfLoader(ACCT_STRING, FIXTURES_DF_STRING)
    standings_df = kaggleDfLoader(ACCT_STRING, STANDINGS_DF_STRING)
    stats_per_fixture_df = kaggleDfLoader(ACCT_STRING, TEAM_STATS_PER_FIXTURE)
    key_events_df = kaggleDfLoader(ACCT_STRING, KEY_EVENTS_STRING)
    key_event_key = kaggleDfLoader(ACCT_STRING, KEY_EVENT_KEY_DF_STRING)


    # loading descriptive statistics
    describe_choice = questionary.select(
        "Would you like descriptive statistics of the dataset to be presented?",
        choices=["yes", "no"]
        ).ask()
    if describe_choice == "yes":
        dfDescriber(standings_df)


    """Important characteristics that we can grab from the datasets above include Goals, Shots, Penalties, Fouls, 
    Offsides, clearances, crosses, number of corners, number of substitutions, number of saves, yellow cards, and red cards 
    all before half time to be used to make predictions
    """

    # joining each of the datasets together to create a training set
    # lets start with the key_events_df and key_events_key
    tester_df = key_events_df.merge(key_event_key, on="keyEventTypeId", how="inner")

    # join the home team and away team ids with the new df
    tester_df = tester_df.merge(fixtures_df[["eventId", "homeTeamId", "awayTeamId"]], on="eventId", how="inner")

    # join the teams with their ids
    tester_df = tester_df.merge(teams_df[["teamId", "name"]], on="teamId", how="inner")
    tester_df = tester_df.rename(columns={"name": "teamName"})

    # standings_df[["updateDate", "updateTime"]] = standings_df["timeStamp"].str.split(" ", expand=True)
    # tester_df[["updateDate", "updateTime"]] = tester_df["updateDateTime"].str.split(" ", expand=True)

    # create a column in the tester df that shows the number of times that a team has played a game for joining with standings df
    indexes = [0]
    for i in range(1, len(tester_df)):
        if (tester_df["keyEventOrder"][i] < tester_df["keyEventOrder"][i-1]) and (tester_df["keyEventOrder"][i] < 3):
            indexes.append(i)

    team_dict = {team: 0 for team in tester_df["teamName"]}
    final_col_vals = []
    for i in range(0, len(indexes)):
        ind_list = []
        if i != len(indexes)-1:
            for team in tester_df.loc[indexes[i]:indexes[i+1]-1, "teamName"]:
                if team not in ind_list:
                    ind_list.append(team)
                    team_dict[team] += 1
                final_col_vals.append(team_dict[team])

        else:
            for team in tester_df.loc[indexes[i]:, "teamName"]:
                if team not in ind_list:
                    ind_list.append(team)
                    team_dict[team] += 1
                final_col_vals.append(team_dict[team])

    tester_df["gamesPlayed"] = final_col_vals
    print(list(set(tester_df["gamesPlayed"])))

    # join each teams standing with their row in the df based off of three characteristics
    excluded_cols = ["year", "last_matchDateTime", "next_opponent", "next_homeAway", "next_matchDateTime"]
    tester_df = tester_df.merge(standings_df.loc[:, ~standings_df.columns.isin(excluded_cols)], on=["gamesPlayed", "seasonType", "teamId"])
    print(list(set(standings_df["gamesPlayed"])))

    print(tester_df.head())
    print(tester_df.columns)
    print(tester_df.shape)
    print(list(set(tester_df["keyEventName"])))

    # tester_df.to_csv("tester.csv", index=False)

    # # preprocess the dataset to get rid of or transform any non-numerical columns
    # df = initial_preprocessDf(df)
    # # check_df = df[df["Date"].str[-2:] == "02"]
    # # print(check_df["match_week"].head())

    # # engineer more useful features
    # dicty = feature_engineer(df)
    # print(dicty["00/01 Season"].iloc[:30, 10:])
    # print(dicty["00/01 Season"].iloc[-30:, 10:])
    # print(dicty["00/01 Season"].dtypes)

    # # final preprocessing to transform string features to numerical ones and to scale features for ml model fitting
    
    # # # seperate the data for training
    # # X_train, X_test, y_train, y_test = data_extract(df)
    # # print(X_train.head(), y_train.head())
    # # print(X_test.head(), y_test.head())


if __name__ == "__main__":
    main()