import pandas as pd
import numpy as np

from copy import deepcopy

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import src.presets as presets

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()

    print("Loading data...")
    train = pd.read_csv('input/train.csv', index_col='id')

    train_X = train['comment_text']
    train_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    cv = KFold(5, shuffle=True, random_state=43)

    scores = pd.DataFrame(data=np.nan, columns=train_y.columns, index=range(cv.n_splits))

    for fold, (fold_train_idx, fold_test_idx) in enumerate(cv.split(range(train.shape[0]))):
        print()
        print("Fold {}:".format(fold))

        fold_model = deepcopy(getattr(presets, args.preset))
        fold_model.fit(train_X.iloc[fold_train_idx], train_y.iloc[fold_train_idx])

        fold_preds = fold_model.predict(train_X.iloc[fold_test_idx])

        scores.loc[fold] = roc_auc_score(train_y.iloc[fold_test_idx], fold_preds, average=None)

        print("  Label scores: {}".format(scores.loc[fold].to_dict()))
        print("  Avg score: {}".format(scores.loc[fold].mean()))

    print("Done.")


if __name__ == "__main__":
    main()
