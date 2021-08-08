#include <pcl/io/pcd_io.h>
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

//typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

void saveDescriptors(pcl::PointCloud<DescriptorType>::Ptr model_descriptors, std::string filename);
void saveKeyPoints(pcl::PointCloud<PointType>::Ptr model_keypoints, std::string filename);

int main (int argc, char *argv[])
{

	if(argc < 4) 
	{
		std::cout << "Incorrect number of parameters: (1) full model path; (2) partial model path; (3) name of the model (without extension);";
		return 0;
	}

	pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr partial_model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr partial_model_keypoints (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<NormalType>::Ptr partial_model_normals (new pcl::PointCloud<NormalType> ());
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
	pcl::PointCloud<DescriptorType>::Ptr partial_model_descriptors (new pcl::PointCloud<DescriptorType> ());

	//pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);

	//
	//  Load clouds
	//
	if (pcl::io::loadPCDFile (argv[1], *model) < 0)
	{
		std::cout << "Error loading model cloud." << std::endl;
		return (-1);
	}
	if (pcl::io::loadPCDFile (argv[2], *partial_model) < 0)
	{
		std::cout << "Error loading model cloud." << std::endl;
		return (-1);
	}

	std::string name = argv[3];
	//
	//  Compute Normals
	//
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch (10);
	// Full model
	norm_est.setInputCloud (model);
	norm_est.compute (*model_normals);
	// Partial Model
	norm_est.setInputCloud (partial_model);
	norm_est.compute (*partial_model_normals);

	//
	//  Downsample Clouds to Extract keypoints
	//
	float model_ss_ = 1.0f;// 0.01f;
	pcl::UniformSampling<PointType> uniform_sampling;
	// Full model
	uniform_sampling.setInputCloud (model);
	uniform_sampling.setRadiusSearch (model_ss_);
	uniform_sampling.filter (*model_keypoints);
	std::cout << name << " total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
	saveKeyPoints(model_keypoints, "../matlab/" + name + "_keypoints.csv");
	// Partial Model
	uniform_sampling.setInputCloud (partial_model);
	uniform_sampling.setRadiusSearch (model_ss_);
	uniform_sampling.filter (*partial_model_keypoints);
	std::cout << "Partial " << name << " total points: " << partial_model->size () << "; Selected Keypoints: " << partial_model_keypoints->size () << std::endl;
	//saveKeyPoints(partial_model_keypoints, "../matlab/partial_" + name + "_keypoints.csv");

	//
	//  Compute Descriptor for keypoints
	//
	// //pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	// //descr_est.setInputReferenceFrames(model_rf);
	float descr_rad_ = 1;// (0.02f);
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
	descr_est.setRadiusSearch (descr_rad_);
	// Full Model
	descr_est.setInputCloud (model_keypoints);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);
	//std::cout << "Descr size: " << model_descriptors->size() << "\n";
	//std::cout << "Descr ref frame: " << *descr_est.getInputReferenceFrames() << "\n End\n";
	//std::cout << "Descr ref frame: " << model_descriptors->points[0] << "\n";
	//saveDescriptors(model_descriptors, "../matlab/" + name + "_descriptors.csv");
	// Partial Model
	descr_est.setInputCloud (partial_model_keypoints);
	descr_est.setInputNormals (partial_model_normals);
	descr_est.setSearchSurface (partial_model);
	descr_est.compute (*partial_model_descriptors);
	saveDescriptors(partial_model_descriptors, "../matlab/partial_" + name + "_descriptors.csv");

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