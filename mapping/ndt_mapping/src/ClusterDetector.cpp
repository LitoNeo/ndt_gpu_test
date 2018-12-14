//
// Created by yunle on 18-10-25.
//

#include <detect/ClusterDetector.h>

#include "../include/detect/ClusterDetector.h"
#include <pcl/features/don.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <cmath>

LidarSegmentation::ClusterDetector::ClusterDetector(const LidarSegmentation::ClusterDetector &) {

}

LidarSegmentation::ClusterDetector &
LidarSegmentation::ClusterDetector::operator=(LidarSegmentation::ClusterDetector & detector) {
	return detector;
}

LidarSegmentation::ClusterDetector::ClusterDetector():_inputCloud(new pcl::PointCloud<PointT>){

}

void LidarSegmentation::ClusterDetector::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
	this->_inputCloud = inputCloud;
}

void LidarSegmentation::ClusterDetector::setDownsample(bool downsample) {
	_downsample = downsample;
}

void LidarSegmentation::ClusterDetector::setLeafSize(float leafSize) {
	_leafsize = leafSize;
}

void LidarSegmentation::ClusterDetector::setRemoveGround(bool removeGround) {
	_removeGround = removeGround;
}

void LidarSegmentation::ClusterDetector::setMinClusterSize(int size) {
	_minClusterSize = size;
}

void LidarSegmentation::ClusterDetector::setMaxClusterSize(int size) {
	_maxClusterSize = size;
}

void LidarSegmentation::ClusterDetector::setMinClipHeight(double minHeight) {
	_minClipHeight = minHeight;
}

void LidarSegmentation::ClusterDetector::setMaxClipHeight(double maxHeight) {
	_maxClipHeight = maxHeight;
}

void LidarSegmentation::ClusterDetector::setKeepLanes(bool keep) {
	_keepLane = keep;
}

void LidarSegmentation::ClusterDetector::setLaneLeftDistance(double leftDistance) {
	_laneLeftBoundary = leftDistance;
}

void LidarSegmentation::ClusterDetector::setLaneRightDistance(double rightDistance) {
	_laneRightBoundary = rightDistance;
}

void LidarSegmentation::ClusterDetector::setClusterMergeThreshold(double thresh) {
	_clusterMergeThresh = thresh;
}

void LidarSegmentation::ClusterDetector::setClusterDistance(double distance) {
	_clusterDistanceThresh = distance;
}

void LidarSegmentation::ClusterDetector::setNoiseDistance(double distance) {
		_noiseDistance = distance;
}

void LidarSegmentation::ClusterDetector::segment() {
	pcl::PointCloud<PointT>::Ptr noiseRemovedCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr downsampledCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr inLanesCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr noFloorCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr onlyFloorCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr diffNormalsCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr clippedCloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusteredPtr(new pcl::PointCloud<pcl::PointXYZRGB>);

	if (_noiseDistance > 0.0) {
		removePointsUpTo(_inputCloud, noiseRemovedCloud);
	} else
		noiseRemovedCloud = _inputCloud;

	if (_downsample) {
		downsample(noiseRemovedCloud, downsampledCloud);
	} else
		downsampledCloud = noiseRemovedCloud;

	clipCloud(downsampledCloud, clippedCloud);

	if (_keepLane) {
		keepLanePoints(clippedCloud, inLanesCloud);
	} else
		inLanesCloud = clippedCloud;

	if (_removeGround) {
		removeFloor(inLanesCloud, noFloorCloud, onlyFloorCloud);
	} else
		noFloorCloud = inLanesCloud;

	differenceNormalsSegmentation(noFloorCloud, diffNormalsCloud);

	segmentByDistance(diffNormalsCloud, clusteredPtr);
}

void LidarSegmentation::ClusterDetector::keepLanePoints(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr outCloud) {
	outCloud->points.clear();
	for (int i = 0; i < inputCloud->points.size(); ++i) {
		auto y = inputCloud->points[i].y;
		if (y <= _laneLeftBoundary && y >= -1.0 * _laneRightBoundary) {
			outCloud->points.push_back(inputCloud->points[i]);
		}
	}
}

void LidarSegmentation::ClusterDetector::segmentByDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr outCloud) {
	std::vector<Cluster::Ptr> allClusters;
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::copyPointCloud<PointT, PointT>(*inputCloud, *cloud);

	clusterAndColor(cloud, outCloud);
}

void LidarSegmentation::ClusterDetector::removeFloor(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                     pcl::PointCloud<pcl::PointXYZ>::Ptr noFloorCloud,
                                                     pcl::PointCloud<pcl::PointXYZ>::Ptr floorCloud) {
	pcl::SACSegmentation<PointT> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	seg.setOptimizeCoefficients(true);
	seg.setMethodType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setAxis(Eigen::Vector3f(0, 0, 1));
	seg.setEpsAngle(_maxFloorAngle);
	seg.setDistanceThreshold(_maxFloorHeight);
	seg.setOptimizeCoefficients(true);
	seg.setInputCloud(inputCloud);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size() == 0) {
		std::cout << "Could not estimate a planar model for input cloud" << std::endl;
	}
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(inputCloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(*noFloorCloud);

	extract.setNegative(false);
	extract.filter(*floorCloud);
}

void LidarSegmentation::ClusterDetector::downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr outCloud) {
	pcl::VoxelGrid<PointT> voxelGrid;
	voxelGrid.setInputCloud(inputCloud);
	voxelGrid.setLeafSize(_leafsize, _leafsize, _leafsize);
	voxelGrid.filter(*outCloud);
}

void LidarSegmentation::ClusterDetector::clipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                   pcl::PointCloud<pcl::PointXYZ>::Ptr outCloud) {
	outCloud->points.clear();
	for (int i = 0; i < inputCloud->points.size(); ++i) {
		if (inputCloud->points[i].z >= _minClipHeight && inputCloud->points[i].z <= _maxClipHeight) {
			outCloud->points.push_back(inputCloud->points[i]);
		}
	}
}

void
LidarSegmentation::ClusterDetector::differenceNormalsSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr outCloud) {
	float small_scale = 0.5;
	float large_scale = 2.0;
	float angle_threshold = 0.5;
	pcl::search::Search<PointT>::Ptr tree;
	if (inputCloud->isOrganized()) {
		tree.reset(new pcl::search::OrganizedNeighbor<PointT>);
	} else {
		tree.reset(new pcl::search::KdTree<PointT>(false));
	}
	tree->setInputCloud(inputCloud);
	pcl::NormalEstimationOMP<PointT, pcl::PointNormal> normalEstimation;
	normalEstimation.setInputCloud(inputCloud);
	normalEstimation.setSearchMethod(tree);
	normalEstimation.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);
	pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);

	normalEstimation.setRadiusSearch(small_scale);
	normalEstimation.compute(*normals_small_scale);

	normalEstimation.setRadiusSearch(large_scale);
	normalEstimation.compute(*normals_large_scale);

	pcl::PointCloud<pcl::PointNormal>::Ptr diffnormals_cloud(new pcl::PointCloud<pcl::PointNormal>);
	pcl::copyPointCloud<PointT, pcl::PointNormal>(*inputCloud, *diffnormals_cloud);

	pcl::DifferenceOfNormalsEstimation<PointT, pcl::PointNormal, pcl::PointNormal> diffnormalsEstimator;
	diffnormalsEstimator.setInputCloud(inputCloud);
	diffnormalsEstimator.setNormalScaleLarge(normals_large_scale);
	diffnormalsEstimator.setNormalScaleSmall(normals_small_scale);
	diffnormalsEstimator.initCompute();
	diffnormalsEstimator.computeFeature(*diffnormals_cloud);

	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>);
	range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, angle_threshold)));

	pcl::ConditionalRemoval<pcl::PointNormal> cond_removal;
	cond_removal.setCondition(range_cond);
	cond_removal.setInputCloud(diffnormals_cloud);

	pcl::PointCloud<pcl::PointNormal>::Ptr diffNormalsCloudFiltered(new pcl::PointCloud<pcl::PointNormal>);
	cond_removal.filter(*diffNormalsCloudFiltered);
	pcl::copyPointCloud<pcl::PointNormal, PointT>(*diffNormalsCloudFiltered, *outCloud);
}

void LidarSegmentation::ClusterDetector::removePointsUpTo(const pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                          pcl::PointCloud<pcl::PointXYZ>::Ptr outCloud) {
	outCloud->points.clear();
	for (int i = 0; i < inputCloud->points.size(); ++i) {
		double originDistance = pow(inputCloud->points[i].x, 2) + pow(inputCloud->points[i].y, 2);
		if (originDistance > _noiseDistance * _noiseDistance) {
			outCloud->points.push_back(inputCloud->points[i]);
		}
	}
}

void LidarSegmentation::ClusterDetector::setMaxFloorHeight(double height) {
	this->_maxFloorHeight = height;
}

void LidarSegmentation::ClusterDetector::setMaxFloorAngle(double angle) {
	_maxFloorAngle = angle;
}

void LidarSegmentation::ClusterDetector::clusterAndColor(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud,
                                                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr outCloud) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*inputCloud, *cloud_2d);
	for (int i = 0; i < cloud_2d->points.size(); ++i) {
		cloud_2d->points[i].z = 0;
	}
	if (cloud_2d->points.size() > 0) {
		tree->setInputCloud(cloud_2d);
	}
	std::vector<pcl::PointIndices> clusterIndices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(_clusterDistanceThresh);
	ec.setMinClusterSize(_minClusterSize);
	ec.setMaxClusterSize(_maxClusterSize);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_2d);
	ec.extract(clusterIndices);

	unsigned int k = 0;
	std::vector<Cluster::Ptr> clusters;
	for(auto it = clusterIndices.begin(); it != clusterIndices.end(); ++ it) {
		Cluster::Ptr cluster(new Cluster);
		//cluster->setCloud(inputCloud, it->indices, k, (int)_colors[k].val[0], (int)_colors[k].val[1],
		// (int)_colors[k].val[2], "", true);
		clusters.push_back(cluster);
		k ++;
	}
//	return clusters;
}

void LidarSegmentation::ClusterDetector::checkClusterMerge(size_t clusterId,
                                                           std::vector<LidarSegmentation::Cluster::Ptr> &clusters,
                                                           std::vector<bool> &visitedClusters,
                                                           std::vector<size_t> &mergeIndices, double mergeThreshold) {
	PointT point = clusters[clusterId]->getCentroid();
	for (auto i = 0; i < clusters.size() ; ++ i) {
		if (i != clusterId && ! visitedClusters[i]) {
			PointT p = clusters[i]->getCentroid();
			double distance = pow(point.x - p.x, 2) + pow(point.y - p.y , 2);
			if (distance <= mergeThreshold * mergeThreshold) {
				visitedClusters[i] = true;
				mergeIndices.push_back(i);
				checkClusterMerge(i, clusters, visitedClusters, mergeIndices, mergeThreshold);
			}
		}
	}
}

void LidarSegmentation::ClusterDetector::mergeClusters(const std::vector<LidarSegmentation::Cluster::Ptr> &inClusters,
                                                       std::vector<LidarSegmentation::Cluster::Ptr> &outClusters,
                                                       std::vector<size_t> mergeIndices, const size_t &idx,
                                                       std::vector<bool> &merged) {
	pcl::PointCloud<pcl::PointXYZRGB> sumCloud;
	pcl::PointCloud<PointT> monoCloud;
	Cluster::Ptr mergedCluster(new Cluster);
	for (auto i = 0; i < mergeIndices.size(); ++ i) {
		sumCloud += *(inClusters[mergeIndices[i]]->getCloud());
		merged[mergeIndices[i]] = true;
	}
	std::vector<int> indices(sumCloud.points.size(), 0);
	for (auto j = 0; j < sumCloud.points.size(); ++j) {
		indices[j] = j;
	}
	if (sumCloud.points.size()> 0) {
		pcl::copyPointCloud(sumCloud, monoCloud);
		// mergedCluster->setCloud(monoCloud.makeShared(), indices, idx, (int)_colors[current_index].val[0], (int)_colors[current_index].val[1],
    //                         (int)_colors[current_index].val[2], "", true);
		outClusters.push_back(mergedCluster);
	}
}

void LidarSegmentation::ClusterDetector::checkAllForMerged(std::vector<LidarSegmentation::Cluster::Ptr> &inClusters,
                                                           std::vector<LidarSegmentation::Cluster::Ptr> &outClusters) {
	std::vector<bool> visitedClusters(inClusters.size(), false);
	std::vector<bool> mergedClusters(inClusters.size(), false);
	size_t index = 0;
	for (size_t i = 0; i < inClusters.size(); ++i) {
		if (!visitedClusters[i]) {
			visitedClusters[i] = true;
			std::vector<size_t> mergeIndices;
			checkClusterMerge(index, inClusters, visitedClusters, mergeIndices, this->_clusterMergeThresh);
			mergeClusters(inClusters, outClusters, mergeIndices, index ++, mergedClusters);
		}
	}
	for (size_t i = 0; i < inClusters.size(); ++ i) {
		if (!mergedClusters[i]) {
			outClusters.push_back(inClusters[i]);
		}
	}
}
