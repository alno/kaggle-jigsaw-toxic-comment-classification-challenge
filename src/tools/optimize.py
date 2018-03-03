import pandas as pd

from sklearn.model_selection import KFold

from hyperopt import fmin, tpe, Trials
from hyperopt.base import JOB_STATE_DONE, STATUS_OK
from hyperopt.utils import coarse_utcnow

from src import meta, presets

import argparse
import json
import os


cv = KFold(3, shuffle=True, random_state=123)


def add_trial_from_json(trials, result):
    params = result['params']
    scores = result['scores']
    tid = len(trials.trials)

    new_result = {'status': STATUS_OK, 'loss': -max(scores)}
    new_result.update(result)

    trials.trials.append({
        'tid': tid,
        'state': JOB_STATE_DONE,
        'result': new_result,
        'misc': {
            'tid': tid,
            'vals': dict((k, [v]) for k, v in params.items()),
            'idxs': dict((k, [tid]) for k in params.keys()),
            'cmd': ('domain_attachment', 'FMinIter_Domain'),
            'workdir': None,
        },
        'spec': None,
        'exp_key': None,
        'book_time': coarse_utcnow(),
        'refresh_time': coarse_utcnow()
    })


def optimize(experiment, search_space, report):
    trials = Trials()

    if os.path.exists(report):
        print("Restoring trials from {}".format(report))
        for line in open(report):
            add_trial_from_json(trials, json.loads(line))
        trials.refresh()

    with open(report, 'a') as report_file:
        def run_experiment(params):
            res = experiment(params)
            report_file.write(json.dumps(res) + '\n')
            report_file.flush()
            return res

        return fmin(
            fn=run_experiment,
            space=search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=100
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()
    preset = getattr(presets, args.preset)

    train_X, train_y, _ = meta.get_input_data(preset)

    # Describe experiment
    def experiment(params):
        print("Running experiment for params {}".format(params))
        histories = []
        for fold, (fold_train_idx, fold_val_idx) in enumerate(cv.split(range(train_X.shape[0]))):
            print("Fold {}:".format(fold))
            model = preset(**params)
            histories.append(model.fit_eval(train_X.iloc[fold_train_idx], train_y.iloc[fold_train_idx], train_X.iloc[fold_val_idx], train_y.iloc[fold_val_idx]))

        mean_history = sum(map(pd.DataFrame.from_records, histories)) / len(histories)
        best_auc = mean_history['roc_auc'].max()

        print("Mean metric history for params {}:\n {}\n".format(params, mean_history))

        return dict(status=STATUS_OK, loss=-best_auc, scores=list(mean_history['roc_auc']), params=params)

    # Run optimization
    best = optimize(experiment, preset.param_search_space, report='hp-report-{}.json'.format(args.preset))

    print("Done, best: {}".format(best))


if __name__ == "__main__":
    main()
