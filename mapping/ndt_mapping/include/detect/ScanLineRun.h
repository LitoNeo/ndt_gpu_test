//
// Created by yunle on 18-11-1.
//

#ifndef FAST_NDT_SLAM_SCANLINERUN_H
#define FAST_NDT_SLAM_SCANLINERUN_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <velodyne_pointcloud/point_types.h>
#include <Eigen/Eigen>

namespace LidarSegmentation {
	struct PointXYZIRL {
		PCL_ADD_POINT4D;
		float intensity;
		uint16_t ring;
		uint16_t label;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};
}
POINT_CLOUD_REGISTER_POINT_STRUCT(LidarSegmentation::PointXYZIRL, (float, x, x)
	(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring)(uint16_t, label, label))

namespace LidarSegmentation {
	using PointT = pcl::PointXYZ;
	class GroundFilter {
	public:
		void setSensorHeight(float height=0.5);

		void setDistThresh(float thresh=0.3);

		void setInitSeedsNumber(int number = 300);

		void setIter(int iter=20);

		void segmentation(pcl::PointCloud<PointT>::ConstPtr input_cloud, pcl::PointIndices &ground_indices, pcl::PointIndices &non_ground_indices);

	protected:

		void extract_initial_seeds_(pcl::PointCloud<PointT>::ConstPtr input_cloud, pcl::PointIndices &ground_indices);

		void estimate_plane_(pcl::PointCloud<PointT>::ConstPtr input_cloud, pcl::PointIndices &ground_indices);

	private:
		float height;
		float dist_thresh;
		int lnr;
		int iter;
		Eigen::Vector3f normal;
		float plane_d;
		Eigen::Vector3f mean;
		Eigen::Matrix3f covariance;
	};
	class LabelSet {
	public:
		typedef boost::shared_ptr<LabelSet> Ptr;
		LabelSet();
		/**
		 * assign new set.
		 * */
		int getNewLabel();
		/**
		 * merge set s and set b
		 * */
		void mergeLabel(int s, int b);
		/**
		 * get the root of set s.
		 * */
		int findRoot(int s);
		/**
		 * increment n elements to set s.
		 * */
		void incrementToSet(int s, int n);
		/**
		 * @brief count points in set s.
		 * */
		int getElementsNumber(int s);

		/**
		 * query all sets' root.
		 * */
		 void getSets(std::vector<std::pair<int, int> > &sets);

	private:
		std::vector<int> forest;
		std::vector<int> number;
		int max_label_;
	};
	/**
	 * segment the point cloud based on Scan Line Run algorithm
	 * Todo: after detecting the clusters, filtering the clusters based on the internal distance to cluster centroid.
	 * **/
	class ScanRunCluster {
	public:
		void segmentation(pcl::PointCloud<PointT>::ConstPtr input_cloud, std::vector<pcl::PointIndicesPtr> &clusters);
		void setRowDistThresh(float row_dist_thresh);
		void setColDistThresh(float col_dist_thresh);
		void setScan(int scan_number);
		void setHorizonResolution(double res);
		void setVerticalResolution(double res);
		void setMinClusterSize(int min_cluster_size);
		void setMaxClusterSize(int max_cluster_size);
	protected:
		void reorganized_cloud(pcl::PointCloud<PointT>::ConstPtr input_cloud);
		void find_runs_(pcl::PointCloud<PointT>::ConstPtr input_cloud, int scan_line_, LabelSet& label_set);
		void update_labels_(pcl::PointCloud<PointT>::ConstPtr input_cloud, int scan_line_, LabelSet& label_set);

	private:
		int scan_number_;
		double horizon_resolution_;
		double vertical_resolution_;
		float row_dist_thresh_;
		float col_dist_thresh_;
		std::vector< std::vector<int> > laser_frame_idx;
		std::vector<int> laser_label;
		std::vector< int> laser_row_idx;
		int min_cluster_size_;
		int max_cluster_size_;
		int max_label;
		int col_size;
	};
}


#endif //FAST_NDT_SLAM_SCANLINERUN_H
