import pandas as pd
import numpy as np

from src import meta

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--features', type=str)
    parser.add_argument('--column', type=str)

    args = parser.parse_args()

    train = pd.read_csv(os.path.join(meta.input_dir, 'train.csv'), index_col='id')

    if args.fold is not None:
        p = meta.get_model_prediction(args.preset, args.fold, 'val')
    else:
        p = pd.concat(meta.get_model_prediction(args.preset, fi, 'val') for fi in range(meta.cv.n_splits))

    p = p.clip(lower=1e-8, upper=1-1e-8)
    y = train.loc[p.index, meta.target_columns]

    if args.features is not None:
        f = pd.read_pickle(os.path.join('cache', 'features', '%s.pickle' % args.features)).loc[p.index]

    columns = meta.target_columns
    if args.column is not None:
        columns = [args.column]
        p = p[columns]
        y = y[columns]

    # Compute logloss for each example
    losses = -(y * np.log(p) + (1-y) * np.log(1-p))
    losses['mean'] = losses.sum(axis=1) / len(columns)

    # Build results table
    results = pd.concat((train.loc[p.index], p.add_suffix('_p'), losses.add_suffix('_loss')), axis=1).sort_values('mean_loss', ascending=False)

    for idx, row in results.iterrows():
        pred_summary = pd.DataFrame([[row[c], row[c + '_p'], row[c + '_loss']] for c in columns], index=columns, columns=['y', 'p', 'loss'])

        print("INDEX: %s" % idx)
        print("LOSS: %.5f" % row['mean_loss'])
        print("COMMENT:\n\n%r" % row['comment_text'])

        if args.features is not None:
            print("\nFEATURES:\n")
            for col in f.columns:
                print("%s: %r" % (col, f.loc[idx, col]))

        print("\nTARGETS:\n%r\n\n" % pred_summary)

        if input("Enter q to exit, anything else to continue: ") == 'q':
            break

        print("\n========================\n")

if __name__ == "__main__":
    main()
