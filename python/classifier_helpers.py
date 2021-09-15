import numpy as np
import pandas as pd
import os
import re

from typing import List, Tuple
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_test_data(
    folder: str, cats: List[str], n_samples: int, descriptor_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[]], dtype=float).reshape(0, 640)
    Y = np.array([], dtype=float).reshape(0)

    for idx, cat in enumerate(cats):
        if n_samples > 1:
            for j in range(n_samples):
                fname = folder + cat + str(j) + "_" + descriptor_type + ".csv"
                # print(fname)
                feature = pd.read_csv(fname, sep=",", header=None)
                feature.reset_index(drop=True, inplace=True)
                arr = feature.to_numpy()
                if descriptor_type.lower() == 'shot':
                    arr = arr[-1, :].reshape(1, 353)
                    assert arr.shape == (1, 353)
                    arr = np.concatenate([arr, np.zeros((1, 640 - 353))], axis=1)
                    assert arr.shape == (1, 640)
                    arr = np.nan_to_num(arr)
                if descriptor_type.lower() == 'esf' and arr.shape[1] == 641:
                    arr = arr[:, :-1]
                arr.reshape(1, 640)
                X = np.concatenate([X, arr])
                Y = np.concatenate([Y, np.array([idx])])
        else:
            raise NotImplemented
            # Do not append number
    print(X.shape, Y.shape)
    assert (X.shape[1] == 640) and (Y.shape == (X.shape[0],))
    return X, Y

def load_training_data(
    folder: str, cats: List[str], descriptor_type: str,
n_samples=1) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[]], dtype=float).reshape(0, 640)
    Y = np.array([], dtype=float).reshape(0)

    if n_samples == 1:
        for idx, cat in enumerate(cats):
            fname = folder + cat + "_" + descriptor_type + ".csv"
            # print(fname)
            feature = pd.read_csv(fname, sep=",", header=None)
            feature.reset_index(drop=True, inplace=True)
            arr = feature.to_numpy()
            if descriptor_type.lower() == 'shot':
                arr = arr[-1, :].reshape(1, 353)
                assert arr.shape == (1, 353)
                arr = np.concatenate([arr, np.zeros((1, 640-353))], axis=1)
                assert arr.shape == (1, 640)
                arr = np.nan_to_num(arr)
            arr.reshape(1, 640)
            print(arr.shape)
            # feature = np.loadtxt(fname, delimiter=',')
            X = np.concatenate([X, arr])
            Y = np.concatenate([Y, np.array([idx])])
    elif n_samples > 1:
        for idx, cat in tqdm(enumerate(cats)):
            for i in range(n_samples):
                fname = folder + cat + str(i).zfill(4) + "_" + descriptor_type + ".csv"
                # print(fname)
                feature = pd.read_csv(fname, sep=",", header=None)
                feature.reset_index(drop=True, inplace=True)
                arr = feature.to_numpy()
                if descriptor_type.lower() == 'shot':
                    arr = arr[-1, :].reshape(1, 353)
                    assert arr.shape == (1, 353)
                    arr = np.concatenate([arr, np.zeros((1, 640 - 353))], axis=1)
                    assert arr.shape == (1, 640)
                    arr = np.nan_to_num(arr)
                if descriptor_type.lower() == 'esf':
                    arr = arr[:, :-1]
                    # print(arr.shape)
                    arr.reshape((1, 640))
                # feature = np.loadtxt(fname, delimiter=',')
                arr.reshape((1, 640))
                X = np.concatenate([X, arr])
                Y = np.concatenate([Y, np.array([idx])])
    print(X.shape, Y.shape)
    return X, Y


# Evaluate against 3, 5, 7 K-NN and kernel SVM
# Return a list of models, and a list of accuracies
def evaluate(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> Tuple[List[str], List[float]]:
    models = ["3-NN, 5-NN, 7-NN", "SVN"]
    accuracies = []
    for k in [
        3,
        5,
        7,
    ]:
        print("Performing K-NN: ", k)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        ypredict = model.predict(x_test)
        accuracy = (ypredict == y_test).mean()
        accuracies.append(accuracy)

    print("Performing SVN", k)
    # SVN with radial basis function
    model = SVC()  # RBF kernel is default
    model.fit(x_train, y_train)
    ypredict = model.predict(x_test)
    accuracy = (ypredict == y_test).mean()
    accuracies.append(accuracy)
    assert len(accuracies) == 4
    return models, accuracies


def load_data(dataset_dir: str, re_pattern: str, d_length)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    categories = []
    X_train = np.array([[]], dtype=float).reshape(0, d_length)
    Y_train = np.array([], dtype=float).reshape(0)
    X_test = np.array([[]], dtype=float).reshape(0, d_length)
    Y_test = np.array([], dtype=float).reshape(0)
    print("walking dir: ", dataset_dir)
    for dirpath, _, filenames in os.walk(dataset_dir, topdown=True):
        print(dirpath, filenames)
        # if there are files
        if filenames:
            dir_split = dirpath.split('/')
            train_test = dir_split[-1] #train or test
            print(train_test, dirpath, filenames)
            assert train_test in ['train', 'test']
            object_name = dir_split[-2]
            if object_name not in categories:
                print(object_name, dirpath)
                categories.append(object_name)
            for filename in filenames:
                print(filename)
                if re.search(re_pattern, filename):
                    # load and append
                    print(filename)
                    arr = np.loadtxt(dirpath + '/' + filename, delimiter=",", usecols=range(d_length))
                    print(arr.shape)
                    arr = arr.reshape(1,d_length)
                    assert arr.shape == (1, d_length)
                    # arr.reset_index(drop=True, inplace=True)
                    print(arr.shape)

                    idx = categories.index(object_name)
                    print(object_name, idx)
                    if train_test == 'train':
                        X_train = np.concatenate([X_train, arr])
                        Y_train = np.concatenate([Y_train, np.array([idx])]) 
                    elif train_test == 'test':
                        X_test = np.concatenate([X_test, arr])
                        Y_test = np.concatenate([Y_test, np.array([idx])]) 
    
    return X_train, Y_train, X_test, Y_test, categories
            
                



        
if __name__ == '__main__':
    print("hello")
    X_train, Y_train, X_test, Y_test, cats = load_data("/Users/chrisparsons/Desktop/pcl_test/descriptors/practice_dataset/", "esf", d_length=640)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, cats)
    print(cats)
    print(Y_test)