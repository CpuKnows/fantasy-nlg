from itertools import chain
import numpy as np
import pandas as pd


def get_teams(teams_file):
    with open(teams_file) as f:
        teams_dict = eval(f.read())
        teams = [v for v in teams_dict.values()]
        teams = list(chain(*teams))

    id_team_dict = dict()
    id_val = 0
    for k, v in teams_dict.items():
        id_team_dict[id_val] = v
        id_val += 1

    team_id_dict = dict()
    for k, v in id_team_dict.items():
        for v_i in v:
            team_id_dict[v_i] = k

    return teams, id_team_dict, team_id_dict


def get_players(news_file):
    players = pd.read_csv(news_file)
    players_list = players['player'].tolist()
    # Last names
    players_list.extend([i[-1] for i in players['player'].str.split(' ')])
    return set(players_list)


def create_news_stats_dataset(news_file, stats_file, output_file=None):
    """Only works with FootballDB.com stats"""
    news = pd.read_csv(news_file)

    fbdb_stats = pd.read_csv(stats_file)
    fbdb_stats['date'] = pd.to_datetime(fbdb_stats['date'])
    # Don't include preseason games
    fbdb_stats = fbdb_stats.loc[lambda df: df['date'] >= '2018-09-06']

    # NFL weeks start on Thursday
    nfl_week = pd.DataFrame({'week_period': pd.period_range('2018-09-06', periods=17, freq='W-WED'),
                             'week': np.arange(1, 18)})

    # Put all player stats and the corresponding news article in one row
    news_stats_df = fbdb_stats.drop(columns=['url', 'team', 'pass_lg', 'rush_lg', 'rec_lg', 'rush_fd', 'rec_fd'])
    news_stats_df['week_period'] = news_stats_df ['date'].dt.to_period('W-WED')
    news_stats_df['game_dow'] = news_stats_df ['date'].dt.day_name()
    news_stats_df['date'] = news_stats_df ['date'].dt.date
    news_stats_df['opp'] = news_stats_df ['opp'].str.upper()
    news_stats_df['opp'] = news_stats_df ['opp'].replace('LA', 'LAR')
    news_stats_df = pd.merge(news_stats_df , nfl_week, how='left', on='week_period')
    news_stats_df['week'] = news_stats_df ['week'].astype(int)
    news_stats_df = news_stats_df .drop(columns=['week_period'])

    temp_df = news.drop(columns=['player_id'])
    temp_df = temp_df.rename(index=str, columns={'player': 'player_name', 'position': 'player_position'})
    temp_df['date'] = pd.to_datetime(temp_df['date']).dt.date
    full_df = pd.merge(news_stats_df, temp_df, on=['player_name', 'player_position', 'date'])

    full_df.to_csv(output_file, index=False)
    return full_df


def create_inverted_news_dict(news_dict, data_cols, team_id_dict, id_team_dict):
    inverted_news_dict = dict()

    for k, v in news_dict.items():
        if k in data_cols and v is not np.nan:
            if (type(v) is np.float64 or type(v) is float) and v % 1 == 0:
                v = int(v)
            k, v = str(k), str(v)

            if k in ['team', 'opp']:
                team_id = team_id_dict[v]
                team_surface_forms = id_team_dict[team_id]
                for team in team_surface_forms:
                    inverted_news_dict[team] = k
            elif v not in inverted_news_dict:
                inverted_news_dict[v] = k
            elif type(inverted_news_dict[v]) is list:
                inverted_news_dict[v] = inverted_news_dict[v] + [k]
            else:
                inverted_news_dict[v] = [inverted_news_dict[v], k]

    return inverted_news_dict
