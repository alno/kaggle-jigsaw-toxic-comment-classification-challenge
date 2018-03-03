import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src import meta, presets

import os
import shutil
import argparse

cv = KFold(10, shuffle=True, random_state=43)


class FoldCache:
    def __init__(self, directory):
        self.directory = directory
        self.val_preds_file = os.path.join(directory, 'pred-val.pickle')
        self.test_preds_file = os.path.join(directory, 'pred-test.pickle')

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

    train_X, train_y, test_X = meta.get_input_data(preset)

    if hasattr(preset, 'submodels'):
        train_X = [train_X]
        for submodel in preset.submodels:
            submodel_val_p = []
            for fold in range(cv.n_splits):
                submodel_val_p.append(pd.read_pickle(os.path.join(meta.cache_dir, submodel, 'fold-%d' % fold, 'pred-val.pickle')))
            submodel_val_p = pd.concat(submodel_val_p).loc[train_X[0].index]
            train_X.append(submodel_val_p)
        train_X = pd.concat(train_X, axis=1)

    scores = pd.DataFrame(data=np.nan, columns=meta.target_columns, index=range(cv.n_splits))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(range(train_X.shape[0]))):
        print()
        print("Fold {}:".format(fold))

        fold_train_X = train_X.iloc[fold_train_idx]
        fold_train_y = train_y.iloc[fold_train_idx]

        fold_val_X = train_X.iloc[fold_val_idx]
        fold_val_y = train_y.iloc[fold_val_idx]

        fold_test_X = test_X

        fold_cache = FoldCache(os.path.join(preset_dir, 'fold-%d' % fold))
        if fold_cache.exists():
            print("Fold already fitted, skipping...")

            # Load predictions
            fold_val_p = pd.read_pickle(fold_cache.val_preds_file)
            fold_test_p = pd.read_pickle(fold_cache.test_preds_file)
        else:
            fold_cache.recreate()
            fold_model = preset()

            # Add stacking features to test dataset
            if hasattr(preset, 'submodels'):
                fold_test_X = [fold_test_X]
                for submodel in preset.submodels:
                    submodel_test_p = pd.read_pickle(os.path.join(meta.cache_dir, submodel, 'fold-%d' % fold, 'pred-test.pickle'))
                    fold_test_X.append(submodel_test_p)
                fold_test_X = pd.concat(fold_test_X, axis=1)

            # Fit the model
            if hasattr(fold_model, 'fit_eval'):
                fold_model.fit_eval(fold_train_X, fold_train_y, fold_val_X, fold_val_y)
            else:
                fold_model.fit(fold_train_X, fold_train_y)

            # Make and save predictions
            fold_val_p = pd.DataFrame(fold_model.predict(fold_val_X), columns=meta.target_columns, index=fold_val_X.index)
            fold_val_p.to_pickle(fold_cache.val_preds_file)

            fold_test_p = pd.DataFrame(fold_model.predict(fold_test_X), columns=meta.target_columns, index=fold_test_X.index)
            fold_test_p.to_pickle(fold_cache.test_preds_file)

        scores.iloc[fold] = roc_auc_score(fold_val_y, fold_val_p, average=None)

        print("  Label scores: {}".format(scores.iloc[fold].to_dict()))
        print("  Avg score: {}".format(scores.iloc[fold].mean()))

    scores['AVG'] = scores.mean(axis=1)

    with pd.option_context('display.width', 160):
        print("\nSCORES:\n\n%r" % scores.rename(index=lambda i: "  %d" % i))
        print("\n%r\n" % pd.DataFrame({'AVG': scores.mean(axis=0), 'STD': scores.std(axis=0)}).T)

    print("Done.")


if __name__ == "__main__":
    main()
