import numpy as np
import open3d as o3d
import os

def convert_to_pcd(arr: np.array, fname: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    o3d.io.write_point_cloud(fname, pcd)

if __name__ == '__main__':
    for (root,dirs,files) in os.walk('datasets', topdown=True):
        print(root, dirs, files)
        for file in files:
            if file[-4:] == '.csv':
                try:
                    arr = np.loadtxt(root + '/' + file , delimiter=',')
                    fout = root + '/' + file[:-4] + '.pcd'
                    convert_to_pcd(arr, fout)
                except:
                    continue
