import argparse
from collections import namedtuple
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from fantasy_nlg.data_utils import create_news_stats_dataset
from fantasy_nlg.spacy_utils import load_spacy_model
from fantasy_nlg.generate_templates import GenerateTemplates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('team_file', help='Glossary of NFL teams', nargs='?', const=1,
                        default='../data/teams_aliases.txt')
    parser.add_argument('player_news', help='News by player and event', nargs='?', const=1,
                        default='../data/player_news.csv')
    parser.add_argument('player_stats', help='Player stats by game', nargs='?', const=1,
                        default='../data/football_db_player_stats.csv')
    parser.add_argument('news_stats_output', help='Output file matching a stat line to a news update', nargs='?',
                        const=1, default='../data/news_and_stats.csv')
    parser.add_argument('intermediate_template_output', help='Output file of unambiguous template matches', nargs='?',
                        const=1, default='../data/intermediate_templates.csv')
    parser.add_argument('final_template_output', help='Output file of disambiguated templates', nargs='?',
                        const=1, default='../data/output_templates.csv')
    parser.add_argument('template_disambiguation_model', help='Pickled disambiguation model', nargs='?',
                        const=1, default='../models/ngram_nb.pkl')
    parser.add_argument('template_chunk_type_model', help='Pickled chunk type model', nargs='?',
                        const=1, default='../models/record_selection_lr.pkl')
    args = parser.parse_args()

    nlp = load_spacy_model(args.team_file, args.player_news)
    news_stats_df = create_news_stats_dataset(args.player_news, args.player_stats, args.news_stats_output)

    template_generator = GenerateTemplates(nlp, args.team_file, vectorizer=None, clf=None)

    print('Processing documents to create training data')
    token_training_set = template_generator.create_training_data(news_stats_df, args.intermediate_template_output)
    # Take out the training data we want to use
    context_ngrams = []
    context_tags = []
    true_tags = []

    for sample in token_training_set:
        # context ngrams
        bigrams = []
        for bigram in sample[1]:
            if len(bigram) != 0:
                bigrams.append(' '.join(bigram))
        context_ngrams.append(bigrams)

        # context tags
        tags = []
        for tag in sample[2]:
            if tag is not None:
                tags.append(tag)
        context_tags.append(tags)

        # True tag
        true_tags.append(sample[-1])

    # Text vectorizers
    def noop(d):
        return d
    ngram_vectorizer = CountVectorizer(lowercase=False, tokenizer=noop, analyzer=noop)
    tag_vectorizer = CountVectorizer(lowercase=False, tokenizer=noop, analyzer=noop)

    X_ngrams = ngram_vectorizer.fit_transform(context_ngrams)
    X_tags = tag_vectorizer.fit_transform(context_tags)

    y_tags = np.array([template_generator.data_col_idx[i] for i in true_tags])
    print('Vectorized data shapes:', X_ngrams.shape, X_tags.shape, y_tags.shape)

    # Tag disambiguation
    ngram_clf = MultinomialNB(alpha=0.5)
    ngram_clf.fit(X_ngrams, y_tags)
    predictions = ngram_clf.predict(X_ngrams)
    print('Context n-gram naive bayes accuracy:', np.sum(predictions == y_tags) / len(y_tags))

    tag_clf = MultinomialNB(alpha=0.5)
    tag_clf.fit(X_tags, y_tags)
    predictions = tag_clf.predict(X_tags)
    print('Context tags naive bayes accuracy:', np.sum(predictions == y_tags) / len(y_tags))

    template_generator.create_prediction_func(ngram_vectorizer, ngram_clf)
    _, chunk_training_dict = template_generator.template_transformer(news_stats_df, args.final_template_output,
                                                                     chunking=True)

    print('Pickling template tag disambiguation model')
    TemplateModel = namedtuple('TemplateModel', ['vectorizer', 'classifier'])
    # Need to copy noop() function from vectorizer when unpickling (see vectorizer args: tokenizer, analyzer)
    with open(args.template_disambiguation_model, 'wb') as f:
        pickle.dump(TemplateModel(ngram_vectorizer, ngram_clf), f)

    # Chunking
    print('Training chunk type selection')
    chunk_training_dict = template_generator.remove_infrequent_chunks(chunk_training_dict)
    record_clf = LogisticRegression(C=10.0, max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)

    record_clf.fit(chunk_training_dict['record']['X'], chunk_training_dict['record']['y'].ravel())
    preds = record_clf.predict(chunk_training_dict['record']['X'])

    print('Chunk type accuracy: {}'.format(f1_score(chunk_training_dict['record']['y'].ravel(),
                                                    preds, average='macro')))

    # Save best performing model
    print('Pickling template chunk type model')
    with open(args.template_chunk_type_model, 'wb') as f:
        pickle.dump(record_clf, f)
