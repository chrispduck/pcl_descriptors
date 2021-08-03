#include <pcl/features/esf.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZ PointType;
typedef pcl::ESFSignature640 DescriptorType;

int main(int argc, char *argv[])
{
    pcl::ESFEstimation<PointType, PointType> ESF;
    if (argc < 5)
    {
        std::cout << "Incorrect number of parameters: \n";
        std::cout << " - arg1: pcd model path; \n";
        std::cout << " - arg2: path to save keypoints; \n";
        std::cout << " - arg3: path to save descriptors; \n";
        std::cout << " - arg4: filename to be saved (without extension) \n";
        return 0;
    }

    pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>());

    //
    //  Load clouds
    //
    if (pcl::io::loadPCDFile(argv[1], *model) < 0)
    {
        std::cout << "Error loading model cloud." << std::endl;
        return (-1);
    }

    std::string keypoints_path = argv[2];
    std::string descriptors_path = argv[3];
    std::string name = argv[4];
}