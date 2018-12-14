//
// Created by yunle on 18-10-25.
//

#include <detect/Cluster.h>

#include "../include/detect/Cluster.h"

namespace LidarSegmentation{
	Cluster::Cluster() {
		valid_cluster_ = true;
	}

	void Cluster::setCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_origin_cloud_ptr,
	                       const std::vector<int> &in_cluster_indices, int in_id,
	                       int in_r, int in_g, int in_b, std::string in_label, bool in_estimate_pose) {
		label_ = label_;
		id_ = in_id;
		r_ = in_r;
		g_ = in_g;
		b_ = in_b;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		min_point_.x = std::numeric_limits<float>::max();
		max_point_.x = std::numeric_limits<float>::min();
		min_point_.y = std::numeric_limits<float>::max();
		max_point_.y = std::numeric_limits<float>::min();
		min_point_.z = std::numeric_limits<float>::max();
		max_point_.z = std::numeric_limits<float>::min();

		centroid_.x = 0;
		centroid_.y = 0;
		centroid_.z = 0;

		for (auto it = 0; it < in_origin_cloud_ptr->points.size(); ++it) {
			pcl::PointXYZRGB p;
			p.x = in_origin_cloud_ptr->points[it].x;
			p.y = in_origin_cloud_ptr->points[it].y;
			p.z = in_origin_cloud_ptr->points[it].z;
			p.r = in_r;
			p.g = in_g;
			p.b = in_b;
			centroid_.x += p.x;
			centroid_.y += p.y;
			centroid_.z += p.z;
			current_cluster->points.push_back(p);
			min_point_.x = std::min(min_point_.x, p.x);
			min_point_.y = std::min(min_point_.y, p.y);
			min_point_.z = std::min(min_point_.z, p.z);
			max_point_.x = std::max(max_point_.x, p.x);
			max_point_.y = std::max(max_point_.y, p.y);
			max_point_.z = std::max(max_point_.z, p.z);
		}

		if (in_cluster_indices.size() > 0) {
			centroid_.x /= in_cluster_indices.size();
			centroid_.y /= in_cluster_indices.size();
			centroid_.z /= in_cluster_indices.size();
		}
		average_point_.x = centroid_.x;
		average_point_.y = centroid_.y;
		average_point_.z = centroid_.z;

		// todo estimate pose
	}

	Cluster::~Cluster() {

	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Cluster::getCloud() {
		return this->pointcloud_;
	}

	pcl::PointXYZ Cluster::getMinPoint() {
		return min_point_;
	}

	pcl::PointXYZ Cluster::getMaxPoint() {
		return max_point_;
	}

	pcl::PointXYZ Cluster::getAveragePoint() {
		return average_point_;
	}

	pcl::PointXYZ Cluster::getCentroid() {
		return centroid_;
	}

	double Cluster::getOrientationAngle() {
		return this->orientation_angle;
	}

	float Cluster::getLength() {
		return this->length_;
	}

	float Cluster::getWidth() {
		return this->width_;
	}

	float Cluster::getHeight() {
		return this->height_;
	}

	int Cluster::getId() {
		return this->id_;
	}

	std::string Cluster::getLabel() {
		return this->label_;
	}

	Eigen::Matrix3f Cluster::getEigenVectors() {
		return this->eigen_vectors_;
	}

	Eigen::Vector3f Cluster::getEigenValues() {
		return this->eigen_values_;
	}

	bool Cluster::isValid() {
		return this->valid_cluster_;
	}

	void Cluster::setValidity(bool in_valid) {
		this->valid_cluster_ = in_valid;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster::joinCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr) {
		return pcl::PointCloud<pcl::PointXYZ>::Ptr();
	}

	std::vector<float>
	Cluster::getFpfhDescriptor(const unsigned int &in_ompnum_threads, const double &in_normal_search_radius,
	                           const double &in_fpfh_search_radius) {
		std::vector<float> cluster_fpfh(33, 0.0);
		pcl::search::KdTree<pcl::PointXYZRGB>::Ptr norm_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
		if (pointcloud_->points.size() > 0) {
			norm_tree->setInputCloud(pointcloud_);
		}
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normalEstimationOMP;
		normalEstimationOMP.setNumberOfThreads(in_ompnum_threads);
		normalEstimationOMP.setInputCloud(pointcloud_);
		normalEstimationOMP.setSearchMethod(norm_tree);
		normalEstimationOMP.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

		normalEstimationOMP.setRadiusSearch(in_normal_search_radius);
		normalEstimationOMP.compute(*normals);

		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_histograms(new pcl::PointCloud<pcl::FPFHSignature33>);
		pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> fpfh;
		fpfh.setNumberOfThreads(in_fpfh_search_radius);
		fpfh.setInputCloud(pointcloud_);
		fpfh.setInputNormals(normals);
		fpfh.setSearchMethod(norm_tree);
		fpfh.setRadiusSearch(in_fpfh_search_radius);
		fpfh.compute(*fpfh_histograms);

		float fpfh_max = std::numeric_limits<float>::min();
		float fpfh_min = std::numeric_limits<float>::max();

		for (int i = 0; i < fpfh_histograms->size(); ++i) {
			for (int j = 0; j < cluster_fpfh.size(); ++j) {
				cluster_fpfh[j] = cluster_fpfh[j] + fpfh_histograms->points[i].histogram[j];
				fpfh_min = std::min(fpfh_min, cluster_fpfh[j]);
				fpfh_max = std::max(fpfh_max, cluster_fpfh[j]);
			}
			float fpfh_dif = fpfh_max - fpfh_min;
			for (int j = 0; fpfh_dif > 0 && j < cluster_fpfh.size(); j ++) {
				cluster_fpfh[j] = (cluster_fpfh[j] - fpfh_min) / fpfh_dif;
			}
		}
		return cluster_fpfh;
	}

}