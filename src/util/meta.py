import os
import pandas as pd


input_dir = os.getenv('INPUT_DIR', 'input')
cache_dir = os.getenv('CACHE_DIR', 'cache')

input_columns = ['comment_text']
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def input_file(filename):
    return os.path.join(input_dir, filename)


def read_input_data():
    print("Loading data...")

    train = pd.read_csv(input_file('train.csv'), index_col='id')
    test = pd.read_csv(input_file('test.csv'), index_col='id')

    train_X = train[input_columns]
    train_y = train[target_columns]

    test_X = test[input_columns]

    return train_X, train_y, test_X
