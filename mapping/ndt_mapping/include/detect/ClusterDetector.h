//
// Created by yunle on 18-10-25.
//

#ifndef FAST_NDT_SLAM_CLUSTERDETECTOR_H
#define FAST_NDT_SLAM_CLUSTERDETECTOR_H

#include "Cluster.h"
namespace LidarSegmentation {
	class ClusterDetector {
	private:
		ClusterDetector(const ClusterDetector &);

		ClusterDetector &operator=(ClusterDetector &);

	public:
		typedef pcl::PointXYZ PointT;
		typedef boost::shared_ptr<ClusterDetector> Ptr;

		ClusterDetector();

		void setInputCloud(pcl::PointCloud<PointT>::Ptr inputCloud);

		void setDownsample(bool downsample=false);

		void setLeafSize(float leafSize=0.1);

		void setRemoveGround(bool removeGround=true);

		void setMinClusterSize(int size=20);

		void setMaxClusterSize(int size=100000);

		void setMinClipHeight(double minHeight=-1.3);

		void setMaxClipHeight(double maxHeight=0.5);

		void setKeepLanes(bool keep=false);

		void setLaneLeftDistance(double leftDistance=5.0);

		void setLaneRightDistance(double rightDistance=5.0);

		void setClusterMergeThreshold(double thresh=1.5);

		void setClusterDistance(double distance=0.75);

		void setNoiseDistance(double distance=0.1);

		void setMaxFloorHeight(double height=0.5);

		void setMaxFloorAngle(double angle=1.0);

		void segment();

	protected:

		void clusterAndColor(pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outCloud);

		void keepLanePoints(pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outCloud);

		void segmentByDistance(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr outCloud);

		void removeFloor(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr noFloorCloud, pcl::PointCloud<PointT>::Ptr floorCloud);

		void downsample(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outCloud);

		void clipCloud(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outCloud);

		void differenceNormalsSegmentation(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outCloud);

		void removePointsUpTo(const pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outCloud);

		void checkClusterMerge(size_t clusterId, std::vector<Cluster::Ptr>& clusters, std::vector<bool> &visitedClusters, std::vector<size_t>&mergeIndices, double mergeThreshold);

		void mergeClusters(const std::vector<Cluster::Ptr>& inClusters, std::vector<Cluster::Ptr> &outClusters,
				std::vector<size_t> mergeIndices, const size_t& idx, std::vector<bool> &visited);

		void checkAllForMerged(std::vector<Cluster::Ptr>&inClusters, std::vector<Cluster::Ptr>&outClusters);

	private:
		pcl::PointCloud<PointT>::Ptr _inputCloud;
		bool _downsample;
		double _leafsize;
		bool _removeGround;
		int _minClusterSize;
		int _maxClusterSize;
		double _minClipHeight;
		double _maxClipHeight;
		bool _keepLane;
		double _laneLeftBoundary;
		double _laneRightBoundary;
		double _clusterMergeThresh;
		double _clusterDistanceThresh;

		double _noiseDistance;

		double _maxFloorHeight;
		double _maxFloorAngle;

	};
}


#endif //FAST_NDT_SLAM_CLUSTERDETECTOR_H
