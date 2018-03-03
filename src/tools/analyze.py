import pandas as pd
import numpy as np

import src.util.meta as meta

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')
    parser.add_argument('--fold', type=int)

    args = parser.parse_args()

    train = pd.read_csv(os.path.join(meta.input_dir, 'train.csv'), index_col='id')

    p = pd.read_pickle(os.path.join(meta.cache_dir, args.preset, 'fold-{}'.format(args.fold), 'pred-val.pickle'))
    y = train.loc[p.index, meta.target_columns]

    # Compute logloss for each example
    losses = -(y * np.log(p) + (1-y) * np.log(1-p))
    losses['mean'] = losses.sum(axis=1) / len(losses.columns)

    # Build results table
    results = pd.concat((train.loc[p.index], p.add_suffix('_p'), losses.add_suffix('_loss')), axis=1).sort_values('mean_loss', ascending=False)

    for idx, row in results.iterrows():
        pred_summary = pd.DataFrame([[row[c], row[c + '_p'], row[c + '_loss']] for c in meta.target_columns], index=meta.target_columns, columns=['y', 'p', 'loss'])

        print("LOSS: %.5f\n" % row['mean_loss'])
        print("COMMENT:\n\n%r\n" % row['comment_text'])
        print("TARGETS:\n%r\n\n" % pred_summary)

        if input("Enter q to exit, anything else to continue: ") == 'q':
            break

        print("\n========================\n")

if __name__ == "__main__":
    main()
