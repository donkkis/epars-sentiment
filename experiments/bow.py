import os
import mlflow
import preprocess as pr

from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from dotenv import load_dotenv
load_dotenv()

def _preprocess():
    data = pr.read_data()
    data = pr.tokenize_df(data)
    data = pr.strip_stopwords(data)
    data = pr.strip_punctuation(data)
    data = pr.frequent_only(data)
    data = pr.flatten(data)
    data = pr.bag_of_words(data)
    data = pr.to_categorical(data)
    return data

def run():

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    acc_train = []
    acc_test = []

    data = _preprocess()
    X = data.drop(['Sentiment'], axis=1)
    y = data[['Sentiment']]

    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('Fold', i)
        X_train, X_test = X.loc[train_idx, :], X.loc[test_idx, :]
        y_train, y_test = y.loc[train_idx, :], y.loc[test_idx, :]
        model = LinearSVC().fit(X_train, y_train)
        acc_train.append(model.score(X_train, y_train))
        acc_test.append(model.score(X_test, y_test))

    print([f'{a:.2f}' for a in acc_train])
    print([f'{a:.2f}' for a in acc_test])

if __name__ == '__main__':
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment('bagofwords')
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        run()