//
// Created by yunle on 18-11-2.
//

#include "detect/ScanLineRun.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#define PI 3.1415926

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


static void downsamplePoints(const cv::Mat& src, cv::Mat& dst, size_t count)
{
  CV_Assert(count >= 2);
  CV_Assert(src.cols == 1 || src.rows == 1);
  CV_Assert(src.total() >= count);
  CV_Assert(src.type() == CV_8UC3);

  dst.create(1, (int)count, CV_8UC3);
  // TODO: optimize by exploiting symmetry in the distance matrix
  cv::Mat dists((int)src.total(), (int)src.total(), CV_32FC1, cv::Scalar(0));
  if (dists.empty())
    std::cerr << "Such big matrix cann't be created." << std::endl;

  for (int i = 0; i < dists.rows; i++)
  {
    for (int j = i; j < dists.cols; j++)
    {
      float dist = (float)norm(src.at<cv::Point3_<uchar> >(i) - src.at<cv::Point3_<uchar> >(j));
      dists.at<float>(j, i) = dists.at<float>(i, j) = dist;
    }
  }

  double maxVal;
  cv::Point maxLoc;
  minMaxLoc(dists, 0, &maxVal, 0, &maxLoc);

  dst.at<cv::Point3_<uchar> >(0) = src.at<cv::Point3_<uchar> >(maxLoc.x);
  dst.at<cv::Point3_<uchar> >(1) = src.at<cv::Point3_<uchar> >(maxLoc.y);

  cv::Mat activedDists(0, dists.cols, dists.type());
  cv::Mat candidatePointsMask(1, dists.cols, CV_8UC1, cv::Scalar(255));
  activedDists.push_back(dists.row(maxLoc.y));
  candidatePointsMask.at<uchar>(0, maxLoc.y) = 0;

  for (size_t i = 2; i < count; i++)
  {
    activedDists.push_back(dists.row(maxLoc.x));
    candidatePointsMask.at<uchar>(0, maxLoc.x) = 0;

    cv::Mat minDists;
    reduce(activedDists, minDists, 0, CV_REDUCE_MIN);
    minMaxLoc(minDists, 0, &maxVal, 0, &maxLoc, candidatePointsMask);
    dst.at<cv::Point3_<uchar> >((int)i) = src.at<cv::Point3_<uchar> >(maxLoc.x);
  }
}

void generateColors(std::vector<cv::Scalar>& colors, size_t count, size_t factor = 100)
{
  if (count < 1)
    return;

  colors.resize(count);

  if (count == 1)
  {
    colors[0] = cv::Scalar(0, 0, 255);  // red
    return;
  }
  if (count == 2)
  {
    colors[0] = cv::Scalar(0, 0, 255);  // red
    colors[1] = cv::Scalar(0, 255, 0);  // green
    return;
  }

  // Generate a set of colors in RGB space. A size of the set is severel times (=factor) larger then
  // the needed count of colors.
  cv::Mat bgr(1, (int)(count * factor), CV_8UC3);
  randu(bgr, 0, 256);

  // Convert the colors set to Lab space.
  // Distances between colors in this space correspond a human perception.
  cv::Mat lab;
  cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

  // Subsample colors from the generated set so that
  // to maximize the minimum distances between each other.
  // Douglas-Peucker algorithm is used for this.
  cv::Mat lab_subset;
  downsamplePoints(lab, lab_subset, count);

  // Convert subsampled colors back to RGB
  cv::Mat bgr_subset;
  cvtColor(lab_subset, bgr_subset, cv::COLOR_BGR2Lab);

  CV_Assert(bgr_subset.total() == count);
  for (size_t i = 0; i < count; i++)
  {
    cv::Point3_<uchar> c = bgr_subset.at<cv::Point3_<uchar> >((int)i);
    colors[i] = cv::Scalar(c.x, c.y, c.z);
  }
}

std::vector<cv::Scalar> colors;

class ScanLineRunNode {
public:
	using PointT = pcl::PointXYZ;
	void setup(ros::NodeHandle &nh, ros::NodeHandle &private_nh) {
		std::string topic;
		nh.param<std::string>("cloud_points", topic, "/velodyne_points");
		points_sub = nh.subscribe(topic, 5, &ScanLineRunNode::pointcloud_callback, this);
		float sensor_height;
		nh.param<float>("sensorHeight", sensor_height, 3.0);
		groundFilter.setSensorHeight(sensor_height);
		float distGroundThresh;
		nh.param<float>("distGroundThresh", distGroundThresh, 0.5);
		groundFilter.setDistThresh(distGroundThresh);
		int initialSeedNumber;
		nh.param<int>("initSeedNumber", initialSeedNumber, 300);
		groundFilter.setInitSeedsNumber(initialSeedNumber);
		int groundFitIter;
		nh.param<int>("groundFitIter", groundFitIter, 20);
		groundFilter.setIter(groundFitIter);

		float row_dist_thresh;
		nh.param<float>("rowDistThresh", row_dist_thresh, 0.5);
		cluster.setRowDistThresh(row_dist_thresh);
		float col_dist_thresh;
		nh.param<float>("colDistThresh", col_dist_thresh, 0.5);
		cluster.setColDistThresh(col_dist_thresh);
		int scan;
		nh.param<int>("scanNumber", scan, 32);
		cluster.setScan(scan);
		float horizonRes;
		nh.param<float>("horizonResolution", horizonRes, 0.15);
		horizonRes /= 180. * PI;
		cluster.setHorizonResolution(horizonRes);
		float verticalRes;
		nh.param<float>("verticalResolution", verticalRes, 1.25);
		verticalRes /= 180. * PI;
		cluster.setVerticalResolution(verticalRes);
		int minClusterSize;
		nh.param<int>("minClusterSize", minClusterSize, 20);
		cluster.setMinClusterSize(minClusterSize);
		int maxClusterSize;
		nh.param<int>("maxClusterSize", maxClusterSize, 20000);
		cluster.setMaxClusterSize(maxClusterSize);

		ground_point_pub = nh.advertise<sensor_msgs::PointCloud2>("/ground_points", 10);
		clusters_pub = nh.advertise<sensor_msgs::PointCloud2>("/cluster_points", 200);

	}

private:
	void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr &ptr) {
		pcl::PointCloud<PointT>::Ptr input_cloud(new pcl::PointCloud<PointT>());
		pcl::fromROSMsg(*ptr, *input_cloud);
		ROS_INFO("total point size: %d", input_cloud->points.size());
		pcl::PointIndicesPtr ground_indices(new pcl::PointIndices), non_ground_indices(new pcl::PointIndices);
		groundFilter.segmentation(input_cloud, *ground_indices, *non_ground_indices);
		pcl::PointCloud<PointT>::Ptr ground_cloud(new pcl::PointCloud<PointT>);
		pcl::PointCloud<PointT>::Ptr non_ground_cloud(new pcl::PointCloud<PointT>);
		pcl::copyPointCloud(*input_cloud, ground_indices->indices, *ground_cloud);
		pcl::copyPointCloud(*input_cloud, non_ground_indices->indices, *non_ground_cloud);
		ROS_INFO("ground size: %d", ground_cloud->points.size());
		// publish ground points.
		sensor_msgs::PointCloud2 ground_points;
		pcl::toROSMsg(*ground_cloud, ground_points);
		ground_points.header.stamp = ptr->header.stamp;
		ground_points.header.frame_id = ptr->header.frame_id;
		ground_point_pub.publish(ground_points);

		std::vector<pcl::PointIndicesPtr> clusterIndices;
		cluster.segmentation(non_ground_cloud, clusterIndices);
		ROS_INFO("after segmentation cluster number: %d", clusterIndices.size());
		// color cluster

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::PointCloud<pcl::PointXYZRGB> subColoredCloud;
		for (int i = 0; i < clusterIndices.size(); ++i) {
			auto& curClusterIndice = clusterIndices[i];
			cv::Scalar &color = colors[i];
			for (int k = 0; k < curClusterIndice->indices.size(); ++k) {
				auto p = input_cloud->points[curClusterIndice->indices[k]];
				pcl::PointXYZRGB q;
				q.x = p.x;
				q.y = p.y;
				q.z = p.z;
				q.r = color.val[0];
				q.g = color.val[1];
				q.b = color.val[2];
				clusterCloud->points.push_back(q);
			}
		}
		ROS_INFO("after cluster, cluster size: %d", clusterCloud->points.size());
		//publish cluster
		sensor_msgs::PointCloud2 clusterPoints;
		pcl::toROSMsg(*clusterCloud, clusterPoints);
		clusterPoints.header.frame_id = ptr->header.frame_id;
		clusterPoints.header.stamp = ptr->header.stamp;
		clusters_pub.publish(clusterPoints);
	}

private:
	ros::Subscriber points_sub;
	ros::Publisher ground_point_pub;
	ros::Publisher clusters_pub;
	LidarSegmentation::ScanRunCluster cluster;
	LidarSegmentation::GroundFilter groundFilter;
};

int main(int argc, char *argv[]) {
	ros::init(argc, argv, "ScanLineRunCluster");
	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");
	ScanLineRunNode scanLineRunNode;
	::generateColors(colors, 200);
	scanLineRunNode.setup(nh, private_nh);
	ros::spin();
}