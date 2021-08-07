import os
import copy
import pandas as pd
import numpy as np
from scipy import linalg


def compute_centroid_index(filename: str):
    # Assume centroid is (0,0,0) as SHOT descriptor computation centred PCs
    dm_esf = pd.read_csv(filename, header=None)
    centroid = dm_esf.median(axis=0)
    dm_esf -= centroid
    # print(dm_esf.median(axis=0))
    assert dm_esf.shape[1] == 3
    l2 = dm_esf.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
    idx = l2.idxmin()
    return idx


def test_compute_centroid():
    assert compute_centroid_index("matrix.txt") == 3


# Load ESF descriptor
def load_ESF(esf_path: str)->np.ndarray:
    dm_esf = pd.read_csv(esf_path, header=None)
    dv_esf = pd.DataFrame(dm_esf.stack())
    assert dv_esf.shape == (640, 1)
    dv_esf.reset_index(drop=True, inplace=True)
    return dv_esf


# Load SHOT descriptor
def load_SHOT(shot_path: str, pc_path)->np.ndarray:
    dm_shot = pd.read_csv(shot_path, header=None)
    idx = compute_centroid_index(pc_path)  # SHOT descriptor closest to the centroid
    dv_closest_shot = dm_shot.iloc[idx, :]
    assert dv_closest_shot.shape == (353,)

    zeros = pd.DataFrame(np.zeros((640 - 353)))
    dv_closest_shot_padded = pd.concat([dv_closest_shot, zeros])
    assert dv_closest_shot_padded.shape == (640, 1)
    dv_closest_shot_padded.reset_index(drop=True, inplace=True)
    return dv_closest_shot_padded


# create CMESF from vector padded shot and ESF
def create_CMESF(shot: np.ndarray, esf: np.ndarray) -> np.ndarray:
    D = pd.concat([esf, shot], axis=1, join='inner')
    assert D.shape == (640, 2)

    # Create Dhat by centring D
    mean = D.mean(axis=0)
    Dhat = D - mean
    assert Dhat.shape == (640, 2)

    arr = Dhat.to_numpy(na_value=0)

    # Perform SVD
    u, s, vh = linalg.svd(arr, full_matrices=True, compute_uv=True)  # unitary array u, singular value vector s,
    # print(u, s, vh)

    # Natales descriptor
    cmesf_descriptor = u[:, 0] * s[0]
    assert cmesf_descriptor.shape == (640,)
    return cmesf_descriptor


def process(esf_path: str, shot_path: str, pc_path: str, cmesf_path: str):
    shot = load_SHOT(shot_path=shot_path, pc_path=pc_path)
    esf = load_ESF(esf_path=esf_path)
    assert esf.shape == (640, 1)
    cmesf = create_CMESF(shot, esf)
    np.savetxt(cmesf_path, cmesf, delimiter=',')


if __name__ == '__main__':
    # test_compute_centroid()

    ESF_PATH = "../descriptors/tactile_split5/filtered/spam0_esf.csv"
    SHOT_PATH = "../descriptors/tactile_split5/filtered/spam0_shot.csv"
    PC_PATH = "../datasets/tactile_split5/filtered/spam0.csv"
    CMESF_PATH = "../descriptors/tactile_split5/filtered/spam0.csv"

    desc_dir = '../descriptors/'
    pcd_dir = 'datasets/'
    for (root, dirs, files) in os.walk(desc_dir, topdown=True):
        print(root, dirs, files)
        # if there are .csv files to process
        while files:
            assert len(files) % 2 == 0
            relpath = os.path.relpath(root, start=desc_dir)
            files = sorted(files)
            shot_name = files.pop()
            esf_name = files.pop()  # type: str
            object_name = shot_name.split('_')[0]  # e.g. baseball1
            shot_path = root + '/' + shot_name
            esf_path = root + '/' + esf_name
            # print("ESF: {0}, SHOT: {1}".format(esf_path, shot_path))
            assert copy.copy(esf_path).replace("esf", "shot") == shot_path
            pc_path = pcd_dir + relpath + '/' + object_name + '.csv'
            cmesf_path = root + '/' + object_name + '_cmesf.csv'
            print("ESF: {0}, SHOT: {1}\nPC {2} CMESF {3}".format(esf_path, shot_path, pc_path, cmesf_path))
            process(esf_path=esf_path, shot_path=shot_path, cmesf_path=cmesf_path, pc_path=pc_path)
