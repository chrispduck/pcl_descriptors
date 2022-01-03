#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char *argv[])
{
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

    // Fill in the cloud data
    pcl::PCDReader reader;
    if (argc < 3)
    {
        std::cout << "Incorrect number of parameters: \n";
        std::cout << " - arg1: pcd model path; \n";
        std::cout << " - arg2: path to save filename; \n";
        return 0;
    }
    std::string fin = argv[1];
    std::string fout = argv[2];

    // Replace the path below with the path where you saved your file
    if (pcl::io::loadPCDFile(argv[1], *cloud) < 0)
    {
        std::cout << "Error loading model cloud." << std::endl;
        return (-1);
    }

    std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
              << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.035f, 0.035f, 0.035f);
    sor.filter(*cloud_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
              << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;


    // MLS filtering
    
    pcl::PCDWriter writer;
    writer.write(fout, *cloud_filtered,
                 Eigen::Vector4f::Zero(), Eigen::Quaternionf::Identity(), false);

    return (0);
}