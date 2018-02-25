import pandas as pd

from sklearn.model_selection import KFold

from hyperopt import fmin, tpe

import src.presets as presets

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()
    preset = getattr(presets, args.preset)

    print("Loading data...")
    train = pd.read_csv('input/train.csv', index_col='id')

    train_X = train[['comment_text']]
    train_y = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    cv = KFold(3, shuffle=True, random_state=123)
    report_file = open('hp-report-{}.json'.format(args.preset), 'w')

    def experiment(params):
        print("Running experiment for params {}".format(params))
        histories = []
        for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(range(train.shape[0]))):
            print("Fold {}:".format(fold))
            model = preset(**params)
            histories.append(model.fit_eval(train_X.iloc[fold_train_idx], train_y.iloc[fold_train_idx], train_X.iloc[fold_val_idx], train_y.iloc[fold_val_idx]))

        mean_history = sum(map(pd.DataFrame.from_records, histories)) / len(histories)
        best_auc = mean_history['roc_auc'].max()

        print("Mean metric history for params {}:\n {}\n".format(params, mean_history))

        report_file.write(json.dumps(dict(params=params, scores=list(mean_history['roc_auc']))) + '\n')
        report_file.flush()

        return -best_auc

    best = fmin(
        fn=experiment,
        space=preset.hp_search_space,
        algo=tpe.suggest,
        max_evals=100
    )

    print(best)

    print("Done.")


if __name__ == "__main__":
    main()
