import numpy as np
import re
from string import Template

from .data_utils import get_teams
from .generate_templates import record_features, field_features


class NewsGenerator:
    def __init__(self, teams_file, record_clf, template_clf):
        self.teams_file = teams_file
        self.data_cols = [
            'player_name', 'player_position',
            'team', 'week', 'game_dow', 'opp', 'away_game', 'team_score', 'opp_score',
            'pass_attempts', 'pass_completions', 'pass_percent', 'pass_yards', 'pass_ya', 'pass_td', 'pass_int',
            'pass_sack', 'pass_rate',
            'rush_attempts', 'rush_yards', 'rush_avg', 'rush_td',
            'receptions', 'rec_yards', 'rec_avg', 'rec_td', 'rec_targets', 'rec_yac'
        ]
        self.data_col_types = [
            'player', 'player',
            'game', 'game', 'game', 'game', 'game', 'game', 'game',
            'passing', 'passing', 'passing', 'passing', 'passing', 'passing', 'passing', 'passing', 'passing',
            'rushing', 'rushing', 'rushing', 'rushing',
            'receptions', 'receptions', 'receptions', 'receptions', 'receptions', 'receptions'
        ]
        self.record_types = [
            'START', 'END', 'NONE', 'player', 'game', 'passing', 'rushing', 'receptions'
        ]
        self.data_col_idx = dict([(v, k) for k, v in enumerate(self.data_cols)])
        self.idx_data_col = dict([(k, v) for k, v in enumerate(self.data_cols)])
        self.data_col_to_type = dict([(k, v) for k, v in zip(self.data_cols, self.data_col_types)])

        self.data_type_to_col = dict()
        for t, c in zip(self.data_col_types, self.data_cols):
            if t not in self.data_type_to_col:
                self.data_type_to_col[t] = [c]
            else:
                self.data_type_to_col[t] = self.data_type_to_col[t] + [c]

        self.teams_dict, self.teams, self.id_team_dict, self.team_id_dict, self.team_abbr_dict = get_teams(
            self.teams_file)

        self.record_clf = record_clf
        self.template_clf = template_clf

    def get_data_col_types(self, text):
        template_regex = re.compile(r'\$\{([_a-z][_a-z0-9]*)\}')
        tags = re.findall(template_regex, text)
        try:
            return self.data_col_to_type[tags[0]]
        except KeyError:
            return None

    def record_type_mask(self, classes):
        record_types = []
        for cls in classes:
            record_types.append(self.get_data_col_types(cls))
        return record_types

    def doc_processing(self, stats_df):
        record_output = []
        template_output = []
        news_output = []
        clf_classes = np.array(self.record_type_mask(self.template_clf.classes_))

        for row in stats_df[self.data_cols].iterrows():
            news_dict = row[1].to_dict()

            # Doc planning
            record_list = ['START']
            count = 0
            while record_list[-1] != 'END' and count < 10:
                count += 1
                features = record_features(record_list, news_dict, self.record_types, self.data_cols)
                record_list.append(self.record_clf.predict(features.reshape(1, -1))[0])
            record_output.append(record_list)

            # Content selection
            template_text = ''
            for rt in record_list:
                if rt == 'START' or rt == 'END':
                    pass
                elif rt == 'player':
                    template_text += '${player_name} '
                else:
                    features = field_features(self.data_type_to_col[rt], news_dict, self.data_cols)
                    probs = self.template_clf.predict_proba(features.reshape(1, -1))[0]
                    record_mask = np.where(clf_classes == rt, 1.0, 0.0)
                    template_text += self.template_clf.classes_[np.argmax(probs * record_mask)]
            template_output.append(template_text)

            # Surface realization
            surface_forms = []
            for k, v in news_dict.items():
                if type(v) is float and v % 1 == 0:
                    # 3.0 --> 3
                    surface_forms.append((k, int(v)))
                elif k == 'team':
                    # Choose mascot name if preceded by "the" else city name
                    try:
                        tag_index = template_text.index('${team}')
                        if tag_index >= 4:
                            preceding_text = template_text[(tag_index - 4):tag_index]
                            if preceding_text == 'the ':
                                surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][2]))
                            else:
                                surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][1]))
                        else:
                            surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][0]))
                    except ValueError:
                        surface_forms.append((k, v))
                elif k == 'opp':
                    # Choose mascot name if preceded by "the" else city name
                    try:
                        tag_index = template_text.index('${opp}')
                        if tag_index >= 4:
                            preceding_text = template_text[(tag_index - 4):tag_index]
                            if preceding_text == 'the ':
                                surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][2]))
                            else:
                                surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][1]))
                        else:
                            surface_forms.append((k, self.teams_dict[self.team_abbr_dict[v]][0]))
                    except ValueError:
                        surface_forms.append((k, v))
                else:
                    surface_forms.append((k, v))
            surface_forms = dict(surface_forms)
            news_output.append(Template(template_text).substitute(surface_forms))

        return record_output, template_output, news_output
