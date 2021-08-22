"""
    Natale et al compare 3,5,7 k-NN and radial basis function kernel SVM
    This script will compute the accuracies for each of these methods for the descriptors supplied as arguments.
"""

import argparse
from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from compute_cmesf import compute_centroid_index
import pandas as pd
from python import classifier_helpers as h

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-dir",
        default="../descriptors/single/vision/real/filtered_chopped/augmented/",
    )
    p.add_argument("--test-dir", default="../descriptors/tactile_split3/filtered/")
    p.add_argument("--descriptor-type", default="cmesf", help="One of [cmesf, esf, shot]")
    p.add_argument(
        "--cats",
        default=[
            "baseball",
            "beer",
            "camera_box",
            "golf_ball",
            "orange",
            "pack_of_cards",
            "rubix_cube",
            "shampoo",
            "spam",
            "tape",
        ],
    )
    p.add_argument(
        "--n-train-samples", default=200, help="Number of test samples per object"
    )
    p.add_argument(
        "--n-test-samples", default=3, help="Number of test samples per object"
    )
    args = p.parse_args()

    assert args.descriptor_type in ["cmesf", "esf", "shot"]

    # Load training data
    print("Loading training data")
    X_train, Y_train = h.load_training_data(
        folder=args.train_dir, cats=args.cats, descriptor_type=args.descriptor_type, n_samples=args.n_train_samples
    )
    print(X_train.shape, Y_train.shape)

    print("Loading test data")
    # Load test data
    X_test, Y_test = h.load_test_data(
        folder=args.test_dir,
        cats=args.cats,
        n_samples=args.n_test_samples,
        descriptor_type=args.descriptor_type,
    )
    print(X_test.shape, Y_test.shape)

    # Standardisation
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # print(scaler.get_params())
    # print(len(scaler.mean_))

    # Train Classifier

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, Y_train)

    # Evaluate Classifier
    classifier_types, accuracies = h.evaluate(
        x_train=X_train, y_train=Y_train, x_test=X_test, y_test=Y_test
    )
    print(classifier_types, accuracies)

    # Save results

    # (optional) plot graph
