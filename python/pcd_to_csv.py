import argparse
import numpy as np
import open3d as o3d 
import os
import tqdm


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--dataset-dir",
        default="practice_dataset",
        help="directory containing .csv files," "and where to put .pcd files",
    )
    args = argp.parse_args()
    print("Producing .pcd files")
    for (root, dirs, files) in tqdm.tqdm(os.walk(args.dataset_dir, topdown=True)):
        print(root, dirs, files)
        for file in files:
            if file[-4:] == ".pcd":
                try:
                    filename = root + "/" + file
                    print("filename in: ", filename)
                    pc = o3d.io.read_point_cloud(filename)
                    arr = np.asarray(pc.points)
                    med = np.median(arr, axis=0)
                    print(med)
                    arr -= med
                    fout = root + "/" + file[:-4] + ".csv"
                    print("filename out: ", fout)
                    np.savetxt(fout, arr, delimiter=',')
                except:
                    continue
