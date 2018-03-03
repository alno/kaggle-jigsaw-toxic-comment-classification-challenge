import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import src.presets as presets
import src.util.meta as meta

import os
import shutil
import argparse

cv = KFold(10, shuffle=True, random_state=43)


class FoldCache:
    def __init__(self, directory):
        self.directory = directory
        self.val_preds_file = os.path.join(directory, 'pred-val.csv')
        self.test_preds_file = os.path.join(directory, 'pred-test.csv')

    def exists(self):
        return all(map(os.path.exists, [self.val_preds_file, self.test_preds_file]))

    def recreate(self):
        if os.path.exists(self.directory):
            print("Some old files exist, removing them...")
            shutil.rmtree(self.directory)
        os.makedirs(self.directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()

    preset_name = args.preset
    preset = getattr(presets, preset_name)
    preset_dir = os.path.join(meta.cache_dir, preset_name)

    train_X, train_y, test_X = meta.read_input_data()

    scores = pd.DataFrame(data=np.nan, columns=train_y.columns, index=range(cv.n_splits))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(range(train_X.shape[0]))):
        print()
        print("Fold {}:".format(fold))

        fold_train_X = train_X.iloc[fold_train_idx]
        fold_train_y = train_y.iloc[fold_train_idx]

        fold_val_X = train_X.iloc[fold_val_idx]
        fold_val_y = train_y.iloc[fold_val_idx]

        fold_cache = FoldCache(os.path.join(preset_dir, 'fold-{}'.format(fold)))
        if fold_cache.exists():
            print("Fold already fitted, skipping...")

            # Load predictions
            fold_val_p = pd.read_csv(fold_cache.val_preds_file, index_col='id')
            fold_test_p = pd.read_csv(fold_cache.test_preds_file, index_col='id')
        else:
            fold_cache.recreate()
            fold_model = preset()

            # Fit the model
            if hasattr(fold_model, 'fit_eval'):
                fold_model.fit_eval(fold_train_X, fold_train_y, fold_val_X, fold_val_y)
            else:
                fold_model.fit(fold_train_X, fold_train_y)

            # Make and save predictions
            fold_val_p = pd.DataFrame(fold_model.predict(fold_val_X), columns=meta.target_columns, index=fold_val_X.index)
            fold_val_p.to_csv(fold_cache.val_preds_file)

            fold_test_p = pd.DataFrame(fold_model.predict(test_X), columns=meta.target_columns, index=test_X.index)
            fold_test_p.to_csv(fold_cache.test_preds_file)

        scores.loc[fold] = roc_auc_score(fold_val_y, fold_val_p, average=None)

        print("  Label scores: {}".format(scores.loc[fold].to_dict()))
        print("  Avg score: {}".format(scores.loc[fold].mean()))

    print("Done.")


if __name__ == "__main__":
    main()
