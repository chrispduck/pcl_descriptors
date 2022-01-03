"""
    Natale et al compare 3,5,7 k-NN and radial basis function kernel SVM
    This script will compute the accuracies for each of these methods for the descriptors supplied as arguments.
"""

import argparse
from sklearn.neighbors import KNeighborsClassifier
import classifier_helpers as h

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--descriptor-dir", default="/Users/chrisparsons/Desktop/pcl_test/descriptors/practice_dataset/")
    p.add_argument("--descriptor-type", default="cmesf", help="One of [cmesf, esf, shot]")
    args = p.parse_args()

    assert args.descriptor_type in ["cmesf", "esf", "shot"]

    # Load data
    print("Loading data")
    X_train, Y_train, X_test, Y_test, cats = h.load_data(dataset_dir=args.descriptor_dir, re_pattern=args.descriptor_type, d_length=640)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, cats)

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
