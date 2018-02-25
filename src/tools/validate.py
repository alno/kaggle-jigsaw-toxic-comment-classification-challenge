import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import src.presets as presets

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()
    preset = getattr(presets, args.preset)

    print("Loading data...")
    train = pd.read_csv('input/train.csv', index_col='id')

    train_X = train[['comment_text']]
    train_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    cv = KFold(5, shuffle=True, random_state=43)

    scores = pd.DataFrame(data=np.nan, columns=train_y.columns, index=range(cv.n_splits))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(range(train.shape[0]))):
        print()
        print("Fold {}:".format(fold))

        fold_train_X = train_X.iloc[fold_train_idx]
        fold_train_y = train_y.iloc[fold_train_idx]

        fold_val_X = train_X.iloc[fold_val_idx]
        fold_val_y = train_y.iloc[fold_val_idx]

        fold_model = preset()

        if hasattr(fold_model, 'fit_eval'):
            fold_model.fit_eval(fold_train_X, fold_train_y, fold_val_X, fold_val_y)
        else:
            fold_model.fit(fold_train_X, fold_train_y)

        fold_val_p = fold_model.predict(fold_val_X)

        scores.loc[fold] = roc_auc_score(fold_val_y, fold_val_p, average=None)

        print("  Label scores: {}".format(scores.loc[fold].to_dict()))
        print("  Avg score: {}".format(scores.loc[fold].mean()))

    print("Done.")


if __name__ == "__main__":
    main()
