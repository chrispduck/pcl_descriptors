import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_test_data(folder: str, cats: List[str], n_samples: int, descriptor_type: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[]], dtype=float).reshape(0, 640)
    Y = np.array([], dtype=float).reshape(0)

    for idx, cat in enumerate(cats):
        if n_samples > 1:
            for j in range(n_samples):
                fname = folder + cat + str(j) + '_' + descriptor_type + ".csv"
                # print(fname)
                feature = pd.read_csv(fname, sep=',', header=None).stack()
                feature.reset_index(drop=True, inplace=True)
                X = np.concatenate([X, feature.to_numpy().reshape(1, 640)])
                Y = np.concatenate([Y, np.array([idx])])
        else:
            raise NotImplemented
            # Do not append number
    print(X.shape, Y.shape)
    assert (X.shape[1] == 640) and (Y.shape == (X.shape[0],))
    return X, Y


def load_training_data(folder: str, cats: List[str], descriptor_type: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[]], dtype=float).reshape(0, 640)
    Y = np.array([], dtype=float).reshape(0)

    for idx, cat in enumerate(cats):
        fname = folder + cat + '_' + descriptor_type + ".csv"
        # print(fname)
        feature = pd.read_csv(fname, sep=',', header=None).stack()
        feature.reset_index(drop=True, inplace=True)

        # feature = np.loadtxt(fname, delimiter=',')
        X = np.concatenate([X, feature.to_numpy().reshape(1, 640)])
        Y = np.concatenate([Y, np.array([idx])])
    return X, Y


# Evaluate against 3, 5, 7 K-NN and kernel SVM
# Return a list of models, and a list of accuracies
def evaluate(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[
    List[str], List[float]]:
    models = ["3-NN, 5-NN, 7-NN", "SVN"]
    accuracies = []
    for k in [3, 5, 7, ]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        ypredict = model.predict(x_test)
        accuracy = (ypredict == y_test).mean()
        accuracies.append(accuracy)

    # SVN with radial basis function
    model = SVC() # RBF kernel is default
    model.fit(x_train, y_train)
    ypredict = model.predict(x_test)
    accuracy = (ypredict == y_test).mean()
    accuracies.append(accuracy)
    assert len(accuracies) == 4
    return models, accuracies
