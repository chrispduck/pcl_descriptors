"""
    Natale et al compare 3,5,7 k-NN and radial basis function kernel SVM
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler


import pandas as pd

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="../descriptors/single/vision/real/filtered_chopped/train/")
    p.add_argument("--test-dir", default="../descriptors/tactile_split3/filtered/")
    p.add_argument("--descriptor-type", default="esf", help="One of [cmesf, esf, shot]")
    p.add_argument("--cats", default=["baseball", "beer", "camera_box", "golf_ball", "orange", "pack_of_cards", "rubix_cube", "shampoo", "spam", "tape"])
    args = p.parse_args()

    assert args.descriptor_type in ["cmesf", "esf", "shot"]

    # Load training data
    # Standardize to zero mean and 1SD
    X_train = np.zeros((0, 640))
    Y_train = np.zeros((0))
    for idx, cat in enumerate(args.cats):
        fname = args.train_dir + cat + '_' + args.descriptor_type + ".csv"
        print(fname)
        # feature = pd.read_csv(fname, sep=',', header=None).stack().to_numpy()
        # feature = np.loadtxt(fname, delimiter=',')
        X_train = np.concatenate([X_train, feature])
        Y_train = np.concatenate([Y_train, np.array([idx])])

    print(X_train, Y_train)
    # Load test data, use above standardisation
    # scaler = StandardScaler()
    # scaler.fit(train)

    # Train Classifier

    # Evaluate Classifier

    # Save results

    # (optional) plot graph


