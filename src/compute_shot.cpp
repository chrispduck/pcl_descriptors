#include <pcl/io/pcd_io.h>
#include <pcl/io/auto_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include<pcl/visualization/histogram_visualizer.h>
#include <fstream>

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

float findRadius(pcl::PointCloud<PointType>::Ptr cloud);
void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename);
void saveFinalDescriptor(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename);
void saveKeyPoints(pcl::PointCloud<PointType>::Ptr model_keypoints, std::string filename);
void invertNormals(pcl::PointCloud<NormalType>::Ptr model_normals);
float centre(pcl::PointCloud<PointType>::Ptr cloud);

float model_ss_ = 1.0f; //0.5f;// 0.01f; // Radius size for downsampling
float descr_rad_ = 1;// (0.02f);

int main (int argc, char *argv[])
{

	if(argc < 4 || argc > 6) 
	{
		std::cout << "Incorrect number of parameters: \n";
		std::cout << " - arg1: pcd model path; \n";
		std::cout << " - arg2: filename to save keypoints; \n";
		std::cout << " - arg3: filename to save descriptors; \n";
		std::cout << " - arg4: radius for downsampling. Default = 1.0f \n";
		std::cout << " - arg5: description radius. Default = 1.0f \n";
		return 0;
	}
	if(argc == 5) model_ss_ = strtod(argv[5], NULL);
	if(argc == 6) descr_rad_ =  strtod(argv[6], NULL);

	pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());

	//
	//  Load clouds
	//
	if (pcl::io::load(argv[1], *model) < 0)
	{
		std::cout << "Error loading model cloud." << std::endl;
		return (-1);
	}

	std::string keypoints_fname = argv[2];
	std::string descriptors_fname = argv[3];
	
	// Centre the poindcloud
	// float radius = centre(model);
	// Put in a point at the origin and use the resultants shot descriptor as the GLOBAL descriptor.
	// std::cout << model->size() << std::endl;
	model->emplace_back(0,0,0);
	// std::cout << model->size() << std::endl;
	// std::cout << model->points[model->size()-1] << std::endl;


	//
	//  Compute Normals
	//
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch(model->size()-1); //Require ALL points as neighbours
	norm_est.setInputCloud (model);
	norm_est.compute (*model_normals);
	invertNormals(model_normals);

	//
	//  Downsample Clouds to Extract keypoints
	//
	// pcl::UniformSampling<PointType> uniform_sampling;
	// // Full model
	// uniform_sampling.setInputCloud (model);
	// uniform_sampling.setRadiusSearch (radius);
	// uniform_sampling.filter (*model_keypoints);
	// std::cout << name << " total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
	// saveKeyPoints(model_keypoints, keypoints_path + "/" + name + "_keypoints.csv");

	//
	//  Compute Descriptor for keypoints
	//
	// //pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	// //descr_est.setInputReferenceFrames(model_rf);
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	// descr_est.();
	// SHOT Does not work with seach method set to k-neighborhood
	float radius = findRadius(model);
	descr_est.setRadiusSearch (radius); // Use radius of maximimum distance from origin + const to ensure all included. 
	// descr_est.setKSearch(model->size()-1);
	descr_est.setInputCloud (model);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);
	// std::cout << "Descr size: " << model_descriptors->size() << "\n";
	//std::cout << "Descr ref frame: " << *descr_est.getInputReferenceFrames() << "\n End\n";
	//std::cout << "Descr ref frame: " << model_descriptors->points[0] << "\n";
	saveFinalDescriptor(model_descriptors, descriptors_fname + ".csv");

	#if 0
	// Visualize
	pcl::visualization::PCLVisualizer viewer ("Visualization");
	int v1 (0);
	int v2 (1);
	viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
	// Add Model
  	viewer.addPointCloud (model, name + "_cloud",v1);
	viewer.addPointCloud (partial_model, "partial_" + name + "_cloud",v2);
	// Add Keypoints for the Full Model
	pcl::visualization::PointCloudColorHandlerCustom<PointType> model_keypoints_color_handler (model_keypoints, 0, 0, 255);
    viewer.addPointCloud (model_keypoints, model_keypoints_color_handler, name + "_keypoints",v1);
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name + "_keypoints",v1);
	// Add Keypoints for the Partial Model
	//pcl::visualization::PointCloudColorHandlerCustom<PointType> model_keypoints_color_handler (partial_model_keypoints, 0, 0, 255);
    viewer.addPointCloud (partial_model_keypoints, model_keypoints_color_handler, "partial_" + name + "_keypoints",v2);
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "partial_" + name + "_keypoints",v2);
	d
	while (!viewer.wasStopped ())
	{
		viewer.spinOnce ();
	}
	#endif
}

void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename)
{
	std::ofstream myfile;
	myfile.open (filename);
	for (int i = 0; i < model_descriptors->size(); i++ )
	{
		for (int j = 0; j < 352; j++ ) myfile << model_descriptors->points[i].descriptor[j] << ",";
		myfile << "\n";
	}
	myfile.close();
}

// Save the last SHOT descriptor computed. Associated with the final point in the pc array.
void saveFinalDescriptor(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename)
{
	std::ofstream myfile;
	myfile.open (filename);

	for (int j = 0; j < 352; j++ ) myfile << model_descriptors->points[model_descriptors->size()-1].descriptor[j] << ",";
	myfile << "\n";
	myfile.close();
}

void saveKeyPoints(pcl::PointCloud<PointType>::Ptr model_keypoints, std::string filename)
{
	std::ofstream myfile;
	myfile.open (filename);
	for (int i = 0; i < model_keypoints->points.size(); i++ )
	{
		myfile << model_keypoints->points[i].x << ",";
		myfile << model_keypoints->points[i].y << ",";
		myfile << model_keypoints->points[i].z << "\n";
	}
	myfile.close();
}

void invertNormals(pcl::PointCloud<NormalType>::Ptr model_normals)
{
    for( int i = 0; i < model_normals->size(); i++ ) 
    {
        model_normals->points[i].normal_x = -model_normals->points[i].normal_x;
        model_normals->points[i].normal_y = -model_normals->points[i].normal_y;
        model_normals->points[i].normal_z = -model_normals->points[i].normal_z;
    }
}

// Centres a pointcloud around the origin
// returns the maximum distance of a point from the origin.
float centre(pcl::PointCloud<PointType>::Ptr cloud){
	//
	// Compute radius
	//

	PointType p;
	float xmax, xmin, ymax, ymin, zmax, zmin;
	p = cloud->points[0];
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

float findRadius(pcl::PointCloud<PointType>::Ptr cloud){
	//
	// Compute radius
	//

	PointType p;
	float xmax, ymax, zmax;
	xmax=0;ymax=0;zmax=0;

	for (int i=0; i<cloud->size(); i++){
		p = cloud->points[i];
		if (abs(p.x) > xmax){
			xmax=p.x;
		}
		if (abs(p.y) > ymax){
			ymax=p.y;
		}
		if (abs(p.z) > zmax){
			zmax=p.z;
		}

	}

	float radius = std::max({xmax, ymax, zmax});
	return radius;
}
