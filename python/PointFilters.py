import numpy as np
import pcl

class MLS_Voxel_Filter:
    """ A class to perform moving least squares and a voxel filtering to an input cloud"""

    def __init__(self, MLS_radius: float, leafsize: float):
        """
        Args:
            MLS_radius: Moving least squares k-nn radius
            leafsize: length of the voxel cube
        """
        self.MLS_radis = MLS_radius
        self.leafsize = leafsize

    def __call__(self, points: np.ndarray):
        # assert points is np.ndarray
        cloud = pcl.PointCloud(points.astype('float32'))
        # Moving least Squares and voxel filtering
        MLS_filtered_cloud = self.apply_MLS(cloud)
        MLS_and_voxel_cloud = self.apply_voxel_filter(MLS_filtered_cloud)
        return MLS_and_voxel_cloud.to_array()

    def apply_MLS(self, cloud: pcl.PointCloud) -> pcl.PointCloud:
        MLS = cloud.make_moving_least_squares()  # type: pcl.MovingLeastSquares
        MLS.set_search_radius(self.MLS_radis)
        return MLS.process()

    def apply_voxel_filter(self, cloud: pcl.PointCloud) -> pcl.PointCloud:
        voxel_filter = cloud.make_voxel_grid_filter()  # type pcl.VoxelGridFilter
        voxel_filter.set_leaf_size(self.leafsize, self.leafsize, self.leafsize)
        return voxel_filter.filter()