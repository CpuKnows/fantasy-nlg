import argparse
import codecs
import itertools
import json
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Football DB scraped json input',
                        default='../../football_db_spider/scraped_data/football_db_player_stats.json')
    parser.add_argument('output_path', help='csv output file', default='../data/football_db_player_stats.csv')
    args = parser.parse_args()

    # Load scraped player stats from json file
    stats = []
    with codecs.open(args.input_path, 'r') as f:
        for line in f:
            stats.append(json.loads(line))

    # Clean and output to csv
    with codecs.open(args.output_path, 'w') as f:
        f.write(
            'url,player_name,player_position,team,date,opp,away_game,win,team_score,opp_score,pass_attempts,' +
            'pass_completions,pass_percent,pass_yards,pass_ya,pass_td,pass_int,pass_lg,pass_sack,pass_rate,' +
            'rush_attempts,rush_yards,rush_avg,rush_lg,rush_td,rush_fd,receptions,rec_yards,rec_avg,rec_lg,' +
            'rec_td,rec_fd,rec_targets,rec_yac\n')

        # passing_headers = ['Date', 'Opp', 'Att', 'Cmp', 'Pct', 'Yds', 'YPA', 'TD', 'Int', 'Lg', 'Sack', 'Rate',
        #                    'Result']
        # rushing_headers = ['Date', 'Opp', 'Att', 'Yds', 'Avg', 'Lg', 'TD', 'FD', 'Result']
        # receiving_headers = ['Date', 'Opp', 'Rec', 'Yds', 'Avg', 'Lg', 'TD', 'FD', 'Tar', 'YAC', 'Result']
        for player in stats:
            passing_stats = False
            rushing_stats = False
            receiving_stats = False

            # A player record could have any combination of passing, rushing, receiving, or noneW
            num_games = 0
            if 'passing' in player:
                num_games = len(player['passing'])
                passing_stats = True
            if 'rushing' in player:
                num_games = len(player['rushing'])
                rushing_stats = True
            if 'receiving' in player:
                num_games = len(player['receiving'])
                receiving_stats = True

            if num_games != 0:
                dates = []
                opps = []
                away = []
                win = []
                team_score = []
                opp_score = []

                for game in player['passing' if passing_stats else ('rushing' if rushing_stats else 'receiving')]:
                    dates.append(game[0])
                    opps.append(game[1][2:] if '@' in game[1] else game[1])
                    away.append(True if '@' in game[1] else False)
                    win.append(True if 'W' in game[-1] else False)
                    team_score.append(game[-1][3:].split('-')[0 if 'W' in game[-1] else 1])
                    opp_score.append(game[-1][3:].split('-')[1 if 'W' in game[-1] else 0])

                if 'passing' in player:
                    passing_values = player['passing']
                else:
                    passing_values = []

                if 'rushing' in player:
                    rushing_values = player['rushing']
                else:
                    rushing_values = []

                if 'receiving' in player:
                    receiving_values = player['receiving']
                else:
                    receiving_values = []

                for idx, (passing, rushing, receiving) in enumerate(
                        itertools.zip_longest(passing_values, rushing_values, receiving_values,
                                              fillvalue=0)):
                    f.write(
                        player['url'] + ',' + player['player'] + ',' + player['position'] + ',' + player['team'] + ',')
                    f.write(
                        str(dates[idx]) + ',' + str(opps[idx]) + ',' + str(away[idx]) + ',' + str(win[idx]) + ',' +
                        str(team_score[idx]) + ',' + str(opp_score[idx]) + ',')

                    if passing_stats:
                        f.write(','.join([str(i) for i in passing[2:-1]]) + ',')
                    else:
                        f.write('0,0,0,0,0,0,0,0,0,0,')

                    if rushing_stats:
                        f.write(','.join([str(i) for i in rushing[2:-1]]) + ',')
                    else:
                        f.write('0,0,0,0,0,0,')

                    if receiving_stats:
                        f.write(','.join([str(i) for i in receiving[2:-1]]))
                    else:
                        f.write('0,0,0,0,0,0,0,0')

                    f.write('\n')
