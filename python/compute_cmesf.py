"""
This script walks a directory and a for each ESF and SHOT descriptor pair, it computes a CMESF descriptor.
This is the descriptor described by Natale et. al., which performs SVD decomposition on a 640x2 matrix of [ESF, SHOT]
The arguments taken are --pointcloud-dir and --descriptor-dir. The CMESF descriptors are placed in the same folder
as the SHOT/ESF descriptors.
"""

import argparse
import os
import re
import tqdm

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
def load_ESF(esf_path: str) -> pd.DataFrame:
    dm_esf = pd.read_csv(esf_path, header=None)
    dv_esf = pd.DataFrame(dm_esf.stack())
    assert dv_esf.shape == (640, 1)
    dv_esf.reset_index(drop=True, inplace=True)
    return dv_esf


# Load SHOT descriptor
def load_SHOT(shot_path: str, pc_path: str) -> pd.DataFrame:
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
def create_CMESF(shot: pd.DataFrame, esf: pd.DataFrame) -> np.ndarray:
    D = pd.concat([esf, shot], axis=1, join="inner")
    assert D.shape == (640, 2)

    # Create Dhat by centring D
    mean = D.mean(axis=0)
    Dhat = D - mean
    assert Dhat.shape == (640, 2)

    arr = Dhat.to_numpy(na_value=0)

    # Perform SVD
    u, s, vh = linalg.svd(
        arr, full_matrices=True, compute_uv=True
    )  # unitary array u, singular value vector s,
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
    np.savetxt(cmesf_path, cmesf, delimiter=",")


if __name__ == "__main__":
    # test_compute_centroid()
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--descriptor-dir",
        type=str,
        default="../descriptors/",
        help="Directory which contains ESF & SHOT and will place output CMESF",
    )
    argp.add_argument(
        "--pointcloud-dir",
        type=str,
        default="datasets/",
        help="Directory which contains csv files",
    )
    argp.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable to toggle printing"
    )
    args = argp.parse_args()

    print("Computing ESF descriptors for descriptors: ", args.descriptor_dir)
    for (root, dirs, files) in tqdm.tqdm(os.walk(args.descriptor_dir, topdown=True)):
        if args.verbose:
            print("root: ", root)
            print(root, dirs, files)
        # if there are .csv files to process
        # Construct the relant paths: ESF, SHOT, PC, CMESF
        relpath = os.path.relpath(root, start=args.descriptor_dir)
        r_shot = re.compile(".*_shot.csv")
        r_esf = re.compile(".*_esf.csv")
        shot_list = list(filter(r_shot.match, files))
        esf_list = list(filter(r_esf.match, files))
        # print(shot_list, esf_list)
        while shot_list and esf_list:
            shot_name = shot_list.pop()  # e.g. baseball1_shot.csv
            object_name = shot_name.split("_")[0]  # e.g. baseball1
            esf_name = object_name + "_esf.csv"  # baseball1_esf.csv
            esf_list.remove(esf_name)  # definitely both present

            shot_path = root + "/" + shot_name
            esf_path = root + "/" + esf_name
            pc_path = args.pointcloud_dir + relpath + "/" + object_name + ".csv"
            cmesf_path = root + "/" + object_name + "_cmesf.csv"
            if args.verbose:
                print(
                    "ESF: {0}, SHOT: {1}\nPC {2} CMESF {3}".format(
                        esf_path, shot_path, pc_path, cmesf_path
                    )
                )

            process(
                esf_path=esf_path,
                shot_path=shot_path,
                cmesf_path=cmesf_path,
                pc_path=pc_path,
            )

        if args.verbose:
            print(shot_list, esf_list)
        assert len(shot_list) == 0 and len(esf_list) == 0
