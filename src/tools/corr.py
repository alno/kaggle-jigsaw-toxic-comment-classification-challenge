import pandas as pd
import numpy as np

from scipy.stats import ks_2samp
from src import meta

import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('preset1')
    parser.add_argument('preset2')

    args = parser.parse_args()

    first_df = pd.concat(meta.get_model_prediction(args.preset1, fi, 'val') for fi in range(meta.cv.n_splits))
    second_df = pd.concat(meta.get_model_prediction(args.preset2, fi, 'val') for fi in range(meta.cv.n_splits))

    report = []
    for col in meta.target_columns:
        col1 = first_df[col]
        col2 = second_df[col]

        ks_stat, ks_p_value = ks_2samp(col1.values, col2.values)

        report.append(dict(
            pearson=col1.corr(col2, method='pearson'),
            kendall=col1.corr(col2, method='kendall'),
            spearman=col1.corr(col2, method='spearman'),
            ks_stat=ks_stat,
            ks_p_value=ks_p_value,
        ))

    print("\n%r\n" % pd.DataFrame.from_records(report, index=meta.target_columns))


if __name__ == "__main__":
    main()
