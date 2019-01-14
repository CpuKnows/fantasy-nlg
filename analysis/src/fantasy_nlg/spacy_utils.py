import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token

from .data_utils import get_players, get_teams


class NFLTeamRecognizer(object):
    """NER for NFL Teams"""
    name = 'nfl_teams'

    def __init__(self, nlp, teams=tuple(), label='NFL_TEAM'):
        if label in nlp.vocab.strings:
            self.label = nlp.vocab.strings[label]
        else:
            nlp.vocab.strings.add(label)
            self.label = nlp.vocab.strings[label]
        patterns = [nlp(team) for team in teams]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('NFL_TEAMS', None, *patterns)

        Token.set_extension('is_nfl_team', default=False)
        Doc.set_extension('has_nfl_team', getter=self.has_nfl_team)
        Span.set_extension('has_nfl_team', getter=self.has_nfl_team)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for _, start, end in matches:
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            for token in entity:
                token._.set('is_nfl_team', True)
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            span.merge()
        return doc

    def has_nfl_team(self, tokens):
        return any([t._.get('is_nfl_team') for t in tokens])


class NFLPlayerRecognizer(object):
    """NER for NFL Players"""
    name = 'nfl_players'

    def __init__(self, nlp, players=tuple(), label='NFL_PLAYER'):
        if label in nlp.vocab.strings:
            self.label = nlp.vocab.strings[label]
        else:
            nlp.vocab.strings.add(label)
            self.label = nlp.vocab.strings[label]
        patterns = [nlp(player) for player in players]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('NFL_PLAYERS', None, *patterns)

        Token.set_extension('is_nfl_player', default=False)
        Doc.set_extension('has_nfl_player', getter=self.has_nfl_player)
        Span.set_extension('has_nfl_player', getter=self.has_nfl_player)

    def __call__(self, doc):
        matches = self.matcher(doc)
        spans = []
        for _, start, end in matches:
            entity = Span(doc, start, end, label=self.label)
            spans.append(entity)
            for token in entity:
                token._.set('is_nfl_player', True)
            doc.ents = list(doc.ents) + [entity]
        for span in spans:
            span.merge()
        return doc

    def has_nfl_player(self, tokens):
        return any([t._.get('is_nfl_player') for t in tokens])


def load_spacy_model(team_file, players_file):
    nlp = spacy.load('en')

    # Teams
    teams = get_teams(team_file)
    teams = teams[0]

    # Players
    player_list = get_players(players_file)

    component = NFLTeamRecognizer(nlp, teams)
    nlp.add_pipe(component, last=True)
    component = NFLPlayerRecognizer(nlp, player_list)
    nlp.add_pipe(component, last=True)
    Token.set_extension('template_tag', default=None)
    Span.set_extension('record_type', default=None)

    return nlp
