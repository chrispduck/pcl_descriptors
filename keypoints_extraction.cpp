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
#include <pcl/features/fpfh_omp.h>

//typedef pcl::PointXYZRGBA PointType;
typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
//typedef pcl::SHOT352 DescriptorType;
typedef pcl::FPFHSignature33 DescriptorType;

void invertNormals(pcl::PointCloud<NormalType>::Ptr model_normals);

int main (int argc, char *argv[])
{

	if(argc < 2) 
	{
		std::cout << "Incorrect number of parameters\n";
		return 0;
	}

	pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());

	//pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);

	//
	//  Load clouds
	//
	if (pcl::io::loadPCDFile (argv[1], *model) < 0)
	{
		std::cout << "Error loading model cloud." << std::endl;
		return (-1);
	}

    //
	//  Compute Normals
	//
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
	norm_est.setKSearch (5);
	// Full model
	norm_est.setInputCloud(model);
    //norm_est.setRadiusSearch(2);
	norm_est.compute (*model_normals);
    invertNormals(model_normals);
    //pcl::fli

	//
	//  Downsample Clouds to Extract keypoints
	//
	float model_ss_ = 1.0f;// 0.01f;
	pcl::UniformSampling<PointType> uniform_sampling;
	// Full model
	uniform_sampling.setInputCloud (model);
	uniform_sampling.setRadiusSearch (model_ss_);
	uniform_sampling.filter (*model_keypoints);
	std::cout << "Total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    //  Compute Descriptor for keypoints
	//
	// //pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
	// //descr_est.setInputReferenceFrames(model_rf);
	float descr_rad_ = 1;// (0.02f);
	//pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    pcl::FPFHEstimationOMP<PointType,NormalType,DescriptorType> descr_est;
	descr_est.setRadiusSearch (descr_rad_);
    descr_est.setInputCloud (model_keypoints);
	descr_est.setInputNormals (model_normals);
	descr_est.setSearchSurface (model);
	descr_est.compute (*model_descriptors);
    std::cout << "Descr size: " << model_descriptors->size() << "\n";
    std::cout << "Descr size: " << model_descriptors->points[0].histogram[0] << "\n";
		
	// Visualize
	pcl::visualization::PCLVisualizer viewer ("Visualization");
	// Add Model
  	viewer.addPointCloud(model, "Cloud");
    viewer.addPointCloudNormals<PointType,NormalType>(model, model_normals,1,0.5);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Cloud");
	// Add Keypoints for the Full Model
	//pcl::visualization::PointCloudColorHandlerCustom<PointType> model_keypoints_color_handler (model_keypoints, 0, 0, 255);
    //viewer.addPointCloud (model_keypoints, model_keypoints_color_handler, "keypoints");
	//viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, + "keypoints");

	while (!viewer.wasStopped ())
	{
		viewer.spinOnce ();
	}

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
