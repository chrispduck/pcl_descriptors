#include <pcl/features/esf.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZ PointType;
typedef pcl::ESFSignature640 DescriptorType;

void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename);
float centre(pcl::PointCloud<PointType>::Ptr cloud);

int main(int argc, char *argv[])
{   
    pcl::ESFEstimation<PointType, DescriptorType> ESF;
    if (argc < 3)
    {
        std::cout << "Incorrect number of parameters: \n";
        std::cout << " - arg1: pcd model path; \n";
        std::cout << " - arg2: path to save filename; \n";
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

    std::string fname = argv[2];

    centre(model);
    ESF.setInputCloud(model);
    ESF.compute(*model_descriptors);

    // std::cout << "Number of K neighbours %" << ESF.getKSearch() << std::endl;
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
            // if ((j+1)%64 == 0){
            //     myfile << "\n";
            // }
        }
		myfile << "\n";
	}
	myfile.close();
}

// Centres a pointcloud around the origin
// returns the maximum distance of a point from the origin.
float centre(pcl::PointCloud<PointType>::Ptr cloud){
	//
	// Compute radius
	//

	PointType p;
	float xmax, xmin, ymax, ymin, zmax, zmin;
	xmax = p.x; xmin = p.x;
	ymax = p.y; ymin = p.y;
	zmax = p.z; zmin = p.z;
	int s = cloud->size();
	// std::cout << s << std::endl;
	for (int i=1; i<s; i++){
		p = cloud->points[i];
		if (p.x > xmax){
			xmax=p.x;
		}
		if (p.y > ymax){
			ymax=p.y;
		}
		if (p.z > zmax){
			zmax=p.z;
		}
		if (p.x < xmin){
			xmin=p.x;
		}
		if (p.y < ymin){
			ymin=p.y;
		}
		if (p.z < zmin){
			zmin=p.z;
		}
	}
	
	// 
	// Centre around origin 
	//
	float xoff = (xmax+xmin)/2;
	float yoff = (ymax+ymin)/2;
	float zoff = (zmax+zmin)/2;
	// std::cout << "offsets: " << xoff << " " << yoff << " " << zoff << std::endl;

	for (int i=0; i<s; i++){
		cloud->points[i].x -= xoff;
		cloud->points[i].y -= yoff;
		cloud->points[i].z -= zoff;
	}

	
	// Find the redius size
	// std::cout << " " << xmax << " " << xmin << " "<< ymax <<" "<< ymin << " " << zmax << " " << zmin << std::endl;
	float radius = std::max({xmax-xmin, ymax-ymin, zmax-zmin})/2;
	// std::cout << "Using a radius of size: " << radius << std::endl;

	return radius;
}