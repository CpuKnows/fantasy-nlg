import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .data_utils import create_news_stats_dataset
from .spacy_utils import load_spacy_model
from .generate_templates import GenerateTemplates


if __name__ == '__main__':
    nlp = load_spacy_model('../data/teams_aliases.txt', '../data/player_news.csv')
    news_stats_df = create_news_stats_dataset('../data/player_news.csv', '../data/football_db_player_stats.csv',
                                              '../data/news_and_stats.csv')

    template_generator = GenerateTemplates(nlp, '../data/teams_aliases.txt', vectorizer=None, clf=None)

    print('Processing documents to create training data')
    token_training_set = template_generator.create_training_data(news_stats_df, '../data/intermediate_templates.csv')
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
    ngram_vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda d: d, analyzer=lambda d: d)
    tag_vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda d: d, analyzer=lambda d: d)

    X_ngrams = ngram_vectorizer.fit_transform(context_ngrams)
    X_tags = tag_vectorizer.fit_transform(context_tags)

    y_tags = np.array([template_generator.data_col_idx[i] for i in true_tags])
    print('Vectorized data shapes:', X_ngrams.shape, X_tags.shape, y_tags.shape)

    ngram_clf = MultinomialNB(alpha=0.5)
    ngram_clf.fit(X_ngrams, y_tags)
    predictions = ngram_clf.predict(X_ngrams)
    print('Context n-gram naive bayes accuracy:', np.sum(predictions == y_tags) / len(y_tags))

    tag_clf = MultinomialNB(alpha=0.5)
    tag_clf.fit(X_tags, y_tags)
    predictions = tag_clf.predict(X_tags)
    print('Context tags naive bayes accuracy:', np.sum(predictions == y_tags) / len(y_tags))

    template_generator.create_prediction_func(ngram_vectorizer, ngram_clf)
    template_generator.template_transformer(news_stats_df, '../data/output_templates.csv')

    print('Pickling model')
    with open('../models/template_generator_ngram_naive_bayes', 'wb') as f:
        pickle.dump((ngram_vectorizer, ngram_clf), f)
