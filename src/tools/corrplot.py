import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src import meta
from fnmatch import fnmatch

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='toxic')
    parser.add_argument('--method', default='spearman')
    parser.add_argument('--exclude', action='append')

    args = parser.parse_args()

    exclusions = ['test_*', ]
    if args.exclude is not None:
        exclusions += args.exclude

    df = None
    for model_name in os.listdir('cache'):
        if model_name == 'features':
            continue

        if any(fnmatch(model_name, ex) for ex in exclusions):
            continue

        try:
            model_preds = pd.concat(meta.get_model_prediction(model_name, fi, 'val') for fi in range(meta.cv.n_splits))

            if df is None:
                df = pd.DataFrame(index=model_preds.index)

            df[model_name] = model_preds[args.label]
        except:
            print("Error loading predictions for %r..." % model_name)

    print("Computing correlations for %r..." % list(df.columns))
    corr = df.corr(method=args.method)

    sns.clustermap(data=corr, metric='correlation')
    plt.show()


if __name__ == "__main__":
    main()
