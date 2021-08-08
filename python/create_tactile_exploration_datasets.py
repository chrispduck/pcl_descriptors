import numpy as np
import os
import argparse
from typing import List
from tqdm import tqdm


def split_cloud(arr: np.ndarray, n_splits) -> List[np.ndarray]:
    n_pts = arr.shape[0]
    # print("npoints: ", n_pts)
    assert arr.shape[1] == 3 and n_pts > 20
    a, _ = divmod(n_pts, n_splits)
    # [print(a*i, a*(i+1)) for i in range(n_splits-1)]
    splits = [arr[0 : a * (i + 1), :] for i in range(n_splits - 1)]
    splits.append(arr)  # last cloud take remainder points
    return splits


def split_and_save(model_name: str, obj_path: str, outdir: str, nsplits: int):
    arr = np.loadtxt(obj_path, delimiter=",")
    splits = split_cloud(arr, n_splits=nsplits)
    split_p = [str(i * 1 / nsplits) for i in range(1, nsplits + 1)]
    fnames_out = [outdir + p + "/" + model_name for p in split_p]
    # print(fnames_out)
    # [print(arr.shape) for arr in splits]
    [os.makedirs(outdir + p + "/", exist_ok=True) for p in split_p]
    [
        np.savetxt(fname=fname, X=arr, delimiter=",")
        for fname, arr in zip(fnames_out, splits)
    ]
    # [ for fname, arr in zip(fnames_out, splits)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--indir", type=str, default="../datasets/single/tactile/real/filtered/"
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="../datasets/tactile_exploration_1split_filtered_v2/",
    )
    # ap.add_argument("--indir", type=str, default="../datasets/tactile_split3/filtered/")
    # ap.add_argument("--outdir",type=str, default="../datasets/tactile_exploration_3split_filtered_v2/")
    ap.add_argument("--n-models", type=int, default=1)
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
    # ap.add_argument("--n-models", type=int, default=1)
    # ap.add_argument("--cats", default=["baseball"])
    ap.add_argument("--nsplits", default=20)
    args = ap.parse_args()

    if args.n_models > 1:
        model_names = [
            cat + str(i) + ".csv" for cat in args.cats for i in range(args.n_models)
        ]
    else:
        model_names = [cat + ".csv" for cat in args.cats]

    for model_name in tqdm(model_names):
        model_path = args.indir + model_name
        split_and_save(model_name, model_path, outdir=args.outdir, nsplits=args.nsplits)
