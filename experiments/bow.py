import os
import mlflow
import logging
import argparse
import preprocess as pr
import numpy as np

from constants import PHRASES, SENTILEX
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("mlflow").setLevel(logging.ERROR)

CLASSIFIER = LinearSVC

def _preprocess():
    data = pr.read_data()
    data['Sentence_orig'] = data['Sentence']
    data = pr.tokenize_df(data)
    data = pr.strip_stopwords(data)
    data = pr.strip_punctuation(data)
    data = pr.frequent_only(data)
    data = pr.flatten(data)
    data = pr.vectorize(data, ngram_range=(1, 1), vectorizer=CountVectorizer)
    data = pr.semantic_feats(data, phrases=PHRASES + SENTILEX)
    data = data.drop('Sentence', axis=1)
    data = data.drop('Sentence_orig', axis=1)
    data = pr.to_categorical(data)
    return data

def run(classifier=None):
    global CLASSIFIER, ARGS
    if not classifier:
        classifier = CLASSIFIER

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    acc_train = []
    acc_test = []

    data = _preprocess()
    X = data.drop(['Sentiment'], axis=1)
    y = data[['Sentiment']]

    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('Fold', i+1)
        X_train, X_test = X.loc[train_idx, :], X.loc[test_idx, :]
        y_train, y_test = y.loc[train_idx, :], y.loc[test_idx, :]
        model = classifier().fit(X_train, y_train)
        acc_train.append(balanced_accuracy_score(model.predict(X_train), y_train))
        acc_test.append(balanced_accuracy_score(model.predict(X_test), y_test))

    print('Mean train accuracy:', np.mean(acc_train))
    print('Mean test accuracy:', np.mean(acc_test))
    if not ARGS.no_log:
        mlflow.log_param('folds_train_acc', [f'{a:.3f}' for a in acc_train])
        mlflow.log_param('folds_test_acc', [f'{a:.3f}' for a in acc_test])
        mlflow.log_artifact(__file__)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_log', action='store_true')
    ARGS = parser.parse_args()

    if ARGS.no_log:
        run()
    else:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment('bagofwords')
        mlflow.sklearn.autolog()
        with mlflow.start_run():
            run()   