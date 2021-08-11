# The aim of this script is to generate N vision samples from a single
# vision point cloud by means of random rotation, scaling and additive
# gaussian white noise

import argparse
import os
import numpy as np
import scipy.spatial
from tqdm import tqdm


def compute_descriptors(x: np.ndarray, noise_sd: int, scaling_factor: int) -> np.ndarray:
    # if x.shape[0] > 1024:
    rot = scipy.spatial.transform.Rotation.random().as_matrix()
    noise = np.random.normal(loc=0, scale=noise_sd, size=x.shape)
    scf = np.random.uniform(low=2 - scaling_factor, high=scaling_factor)
    # print(noise.shape)
    assert noise.shape == x.shape
    assert isinstance(scf, float)
    assert rot.shape == (3, 3)
    Xhat = np.matmul(rot, x.transpose()).transpose() * scf + noise
    return Xhat


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default="../datasets/single/vision/real/filtered_chopped/test/")
    ap.add_argument("--out-dir", type=str, default="../datasets/single/vision/real/filtered_chopped/augmented/")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--noise-sd", type=int, default=0.5)
    ap.add_argument("--scaling-factor", type=int, default=1.03)
    ap.add_argument(
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
    args = ap.parse_args()

    for cat in tqdm(args.cats):
        f_in = args.in_dir + cat + '.csv'
        arr = np.loadtxt(f_in, delimiter=',')
        for i in range(args.n_samples):
            d = compute_descriptors(x=arr, noise_sd=args.noise_sd, scaling_factor=args.scaling_factor)
            fout = args.out_dir + cat + str(i).zfill(4) + '.csv'
            # print(fout)
            np.savetxt(fout, d, delimiter=',')
