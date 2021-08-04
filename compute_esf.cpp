#include <pcl/features/esf.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZ PointType;
typedef pcl::ESFSignature640 DescriptorType;

void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename);

int main(int argc, char *argv[])
{   
    pcl::ESFEstimation<PointType, DescriptorType> ESF;
    if (argc < 4)
    {
        std::cout << "Incorrect number of parameters: \n";
        std::cout << " - arg1: pcd model path; \n";
        std::cout << " - arg2: path to save descriptors; \n";
        std::cout << " - arg3: filename to be saved (without extension) \n";
        return 0;
    }

    pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>());
    pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>());

    //
    //  Load clouds
    //
    if (pcl::io::loadPCDFile(argv[1], *model) < 0)
    {
        std::cout << "Error loading model cloud." << std::endl;
        return (-1);
    }

    std::string descriptors_path = argv[2];
    std::string fname = argv[3];

    ESF.setInputCloud(model);
    ESF.compute(*model_descriptors);

    std::cout << "Number of K neighbours %" << ESF.getKSearch() << std::endl;
    // std::cout << "Search radius " << ESF.getRadiusSearch() << std::endl;
    // std::cout << "Search method " << ESF.getSearchMethod() << std::endl;
    // std::cout << "Search parameter " << ESF.getSearchParameter() << std::endl;

    saveDescriptors(model_descriptors, fname);
}

void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename)
{
	std::ofstream myfile;
	myfile.open (filename);
	for (int i = 0; i < model_descriptors->size(); i++ )
	{
		for (int j = 0; j < 640; j++ ) {
            myfile << model_descriptors->points[i].histogram[j] << ",";
            if ((j+1)%64 == 0){
                myfile << "\n";
            }
        }
		myfile << "\n";
	}
	myfile.close();
}