from src import meta

import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset')

    args = parser.parse_args()

    if not os.path.exists('submissions'):
        os.makedirs('submissions')

    preds = sum([meta.get_model_prediction(args.preset, fold, 'test') for fold in range(meta.cv.n_splits)]) / meta.cv.n_splits
    preds.to_csv(os.path.join('submissions', '%s-%s.csv.gz' % (args.preset, time.strftime('%Y-%m-%d-%H-%M-%S'))), compression='gzip')

if __name__ == "__main__":
    main()
