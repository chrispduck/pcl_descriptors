import argparse
import numpy as np
import open3d as o3d
import os
import tqdm


def convert_to_pcd(arr: np.array, fname: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    o3d.io.write_point_cloud(fname, pcd)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--dataset-dir",
        default="datasets",
        help="directory containing .csv files," "and where to put .pcd files",
    )
    args = argp.parse_args()
    print("Producing .pcd files")
    for (root, dirs, files) in tqdm.tqdm(os.walk(args.dataset_dir, topdown=True)):
        # print(root, dirs, files)
        for file in files:
            if file[-4:] == ".csv":
                try:
                    arr = np.loadtxt(root + "/" + file, delimiter=",")
                    med = np.median(arr, axis=0)
                    print(med)
                    arr -= med

                    fout = root + "/" + file[:-4] + ".pcd"
                    convert_to_pcd(arr, fout)
                except:
                    continue
