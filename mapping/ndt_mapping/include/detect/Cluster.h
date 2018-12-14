//
// Created by yunle on 18-10-25.
//

#ifndef FAST_NDT_SLAM_CLUSTER_H
#define FAST_NDT_SLAM_CLUSTER_H

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <limits.h>
#include <cmath>
#include <chrono>

namespace LidarSegmentation{
	class Cluster {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_;
		pcl::PointXYZ min_point_;
		pcl::PointXYZ max_point_;
		pcl::PointXYZ average_point_;
		pcl::PointXYZ centroid_;
		double orientation_angle;
		float length_, width_, height_;

		std::string label_;
		int id_, r_, g_, b_;
		Eigen::Matrix3f eigen_vectors_;
		Eigen::Vector3f eigen_values_;
		bool valid_cluster_;

	public:
		void setCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_origin_cloud_ptr,
				const std::vector<int>&in_cluster_indices, int in_id, int in_r, int in_g, int in_b,
				std::string in_label, bool in_estimate_pose);
		Cluster();
		virtual ~Cluster();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloud();
		pcl::PointXYZ getMinPoint();
		pcl::PointXYZ getMaxPoint();
		pcl::PointXYZ getAveragePoint();
		pcl::PointXYZ getCentroid();
		double getOrientationAngle();
		float getLength();
		float getWidth();
		float getHeight();
		int getId();
		std::string getLabel();
		Eigen::Matrix3f getEigenVectors();
		Eigen::Vector3f getEigenValues();
		bool isValid();
		void setValidity(bool in_valid);
		pcl::PointCloud<pcl::PointXYZ>::Ptr joinCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr);

		std::vector<float> getFpfhDescriptor(const unsigned int & in_ompnum_threads, const double & in_normal_search_radius,
				const double & in_fpfh_search_radius);
		typedef boost::shared_ptr<Cluster> Ptr;
	};
}

#endif //FAST_NDT_SLAM_CLUSTER_H
