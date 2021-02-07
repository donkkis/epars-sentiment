import os
import string

import pandas as pd

from constants import STOP_WORDS, FREQUENT
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
load_dotenv()

def read_data(path=None, drop_cols=['ID']):
    if not path:
        path = os.getenv('DEFAULT_DATAPATH')
    data = pd.read_excel(path)
    data = data.drop(drop_cols, axis=1)
    return data

def tokenize_df(data, col='Sentence', tokenizer=word_tokenize, to_lower=True):
    """
    Tokenize strings contained in dataframe using the given tokenizer

    Args:
        data: pandas.DataFrame object
        col: The column name in df to be tokenized
        to_lower: Optionally cast each token to lower caser
        tokenizer: The tokenizer function to be used. 
                   Should accept a string and return list of strings

    Returns:
        df: the modified pandas.DataFrame object
    """

    df = data.copy()
    if to_lower:
        df[col] = pd.Series(map(lambda s: s.lower(), df[col]))
    df[col] = pd.Series(map(lambda s: tokenizer(s), df[col]))
    return df    

def strip_stopwords(data, col='Sentence', stop_words=STOP_WORDS):
    """
        Strip stopwords from _tokenized_ dataframes
    """

    df = data.copy()
    df[col] = pd.Series(map(lambda s: [t for t in s if t not in stop_words], df[col]))
    return df

def strip_punctuation(data, col='Sentence'):
    df = data.copy()
    df[col] =  pd.Series(map(lambda s: [t for t in s if t not in string.punctuation], df[col]))
    return df

def frequent_only(data, col='Sentence'):
    df = data.copy()
    df[col] = pd.Series(map(lambda s: [t for t in s if t in FREQUENT], df[col]))
    return df

def flatten(data, col='Sentence'):
    """
        Revert to string representation from tokenized df
    """
    df = data.copy()
    df[col] = pd.Series(map(lambda l: ' '.join(l), df[col]))
    return df

def to_categorical(data, oh_cols=['Positive', 'Negative', 'Neutral']):
    """
        Revert one hot encoding. Useful with some sklearn functions that expect categorical data
    """
    df = data.copy()
    df['Sentiment'] = pd.Series()
    df.loc[df['Positive'] == 1, 'Sentiment'] = 1
    df.loc[df['Negative'] == 1, 'Sentiment'] = 2
    df.loc[df['Neutral'] == 1, 'Sentiment'] = 3   
    df = df.drop(oh_cols, axis=1)
    return df

def bag_of_words(data, col='Sentence'):
    df = data.copy()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[col])
    df = df.drop(col, axis=1)
    X = pd.DataFrame(X.todense())
    df = pd.concat([X, df], axis=1)
    return df