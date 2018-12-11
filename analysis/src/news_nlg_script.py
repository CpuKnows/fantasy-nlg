import argparse
import pandas as pd
import pickle

from fantasy_nlg.news_nlg import NewsGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('team_file', help='Glossary of NFL teams', nargs='?', const=1,
                        default='../data/teams_aliases.txt')
    parser.add_argument('record_model', help='Model for record selection', nargs='?', const=1,
                        default='../models/record_selection_lr.pkl')
    parser.add_argument('template_model', help='Model for template selection', nargs='?', const=1,
                        default='../models/template_selection_knn.pkl')
    parser.add_argument('input_stats', help='Input stats data', nargs='?', const=1,
                        default='../data/template_test_data.csv')
    parser.add_argument('output_text', help='Output text data', nargs='?', const=1,
                        default='../data/text_output.csv')
    args = parser.parse_args()

    # Load stats data
    test_data = pd.read_csv(args.input_stats)

    # Unpickle models
    with open(args.record_model, 'rb') as f:
        record_clf = pickle.load(f)

    with open(args.template_model, 'rb') as f:
        template_clf = pickle.load(f)

    # Perform NLG
    news_generator = NewsGenerator(args.team_file, record_clf, template_clf)
    record_output, template_output, news_output = news_generator.doc_processing(test_data)

    # Output
    output_df = pd.DataFrame({'record': record_output, 'template': template_output, 'news_update': news_output})
    output_df.to_csv(args.output_text, index=False)
