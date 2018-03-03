import os
import inspect
import pandas as pd

from src import features


input_dir = os.getenv('INPUT_DIR', 'input')
cache_dir = os.getenv('CACHE_DIR', 'cache')

input_columns = ['comment_text']
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def input_file(filename):
    return os.path.join(input_dir, filename)


def get_input_data(preset=None):
    print("Loading data...")

    train = pd.read_csv(input_file('train.csv'), index_col='id')
    test = pd.read_csv(input_file('test.csv'), index_col='id')

    train_X = train[input_columns]
    train_y = train[target_columns]

    test_X = test[input_columns]

    del train, test

    if hasattr(preset, 'features'):
        loaded = {'raw': pd.concat((train_X, test_X))}
        all_X = pd.concat([get_feature(n, loaded) for n in preset.features], axis=1)

        train_X = all_X.loc[train_X.index]
        test_X = all_X.loc[test_X.index]

    return train_X, train_y, test_X


def get_feature(name, loaded={}):
    if name in loaded:
        return loaded[name]

    cache_file_path = os.path.join(cache_dir, 'features', name + '.pickle')
    if os.path.exists(cache_file_path):
        loaded[name] = pd.read_pickle(cache_file_path)
        return loaded[name]

    print("Computing feature %r..." % name)
    feature_fn = getattr(features, name)
    dep_names = inspect.getargspec(feature_fn).args

    res = feature_fn(*[get_feature(d, loaded) for d in dep_names])

    if not os.path.exists(os.path.join(cache_dir, 'features')):
        os.makedirs(os.path.join(cache_dir, 'features'))

    res.to_pickle(cache_file_path)
    loaded[name] = res
    return res
