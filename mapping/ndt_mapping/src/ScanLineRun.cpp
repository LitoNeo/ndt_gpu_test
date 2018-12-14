//
// Created by yunle on 18-11-1.
//

#include <detect/ScanLineRun.h>

#include "../include/detect/ScanLineRun.h"

namespace LidarSegmentation {

	class HeightComparatorAscender {
	public:
		HeightComparatorAscender(pcl::PointCloud<PointT>::ConstPtr &inputCloud): cloud(inputCloud){
		}
		bool operator()(int p1, int p2){
			return this->cloud->points[p1].z < this->cloud->points[p2].z;
		}
		pcl::PointCloud<PointT>::ConstPtr cloud;
	};

	void GroundFilter::setSensorHeight(float height) {
		this->height = height;
	}

	void GroundFilter::setDistThresh(float thresh) {
		this->dist_thresh = thresh;
	}

	void GroundFilter::setInitSeedsNumber(int number) {
		this->lnr = number;
	}

	void GroundFilter::setIter(int iter) {
		this->iter = iter;
	}

	void GroundFilter::segmentation(pcl::PointCloud<PointT>::ConstPtr input_cloud, pcl::PointIndices &ground_indices,
	                                pcl::PointIndices &non_ground_indices) {
		extract_initial_seeds_(input_cloud, ground_indices);
		for (int i = 0; i < this->iter; ++ i) {
			estimate_plane_(input_cloud, ground_indices);
			ground_indices.indices.clear();
			non_ground_indices.indices.clear();
			for (size_t j = 0; j < input_cloud->points.size(); ++ j) {
				auto &p = input_cloud->points[j];
				double dot_value = p.x * normal(0) + p.y * normal(0) + p.z * normal(2);
				if (dot_value + plane_d < dist_thresh) {
					ground_indices.indices.push_back(j);
				} else {
					non_ground_indices.indices.push_back(j);
				}
			}
		}
	}

	void GroundFilter::estimate_plane_(pcl::PointCloud<PointT>::ConstPtr input_cloud, pcl::PointIndices &ground_indices) {

			Eigen::Matrix3f cov;
			Eigen::Vector4f mean;
			pcl::computeMeanAndCovarianceMatrix(*input_cloud, ground_indices, cov, mean);
			Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
			normal = (svd.matrixU().col(2));
			Eigen::Vector3f seeds_mean = mean.head<3>();
			plane_d = -(normal.transpose() * seeds_mean)(0, 0);
	}
	/**
	 * we can use sensor height instead.
	 * */
	void GroundFilter::extract_initial_seeds_(pcl::PointCloud<PointT>::ConstPtr input_cloud,
	                                          pcl::PointIndices &ground_indices) {

		HeightComparatorAscender comparator(input_cloud);
		std::vector<size_t > sortedIndices(input_cloud->points.size(), 0);
		for (size_t i = 0; i < input_cloud->points.size(); ++i) {
			sortedIndices[i] = i;
		}
		std::sort(sortedIndices.begin(), sortedIndices.end(), comparator);
		ground_indices.indices.clear();
		float mean_value = 0;
		int cnt = 0;
		for (size_t j = 0; j < this->lnr && j < input_cloud->points.size(); ++ j) {
			mean_value += input_cloud->points[sortedIndices[j]].z;
			cnt += 1;
		}
		mean_value = cnt == 0 ? 0 : mean_value / cnt;
		for (int k = 0; k < sortedIndices.size(); ++k) {
			size_t idx = sortedIndices[k];
			if (input_cloud->points[idx].z < mean_value + dist_thresh) {
				ground_indices.indices.push_back(idx);
			}
		}
	}

	static int calc_velodyne_16(double angle) {
		angle += 0.2617993877991494;
		int scan = int(angle / 0.032724923474893676);
		return scan;
	}

	static int calc_velodyne_32(double angle) {
		angle += 0.5235987755982988;
		int scan = int(angle/0.02181661564992912);
		return scan;
	}

	static int calc_scan(int scan_number, double angle) {
		if (scan_number == 16) {
			return calc_velodyne_16(angle);
		}
		if (scan_number == 32) {
			return calc_velodyne_32(angle);
		}
	}
#define PI 3.141592653589793

	static int calc_col(double horizon_res, double theta) {
		return static_cast<int>(floor((theta + PI)/horizon_res + 0.5));
	}

	void ScanRunCluster::setRowDistThresh(float row_dist_thresh) {
		row_dist_thresh_ = row_dist_thresh;
	}

	void ScanRunCluster::setColDistThresh(float col_dist_thresh) {
		col_dist_thresh_ = col_dist_thresh;
	}

	void ScanRunCluster::setScan(int scan_number) {
		scan_number_ = scan_number;
	}

	void ScanRunCluster::setHorizonResolution(double res) {
		horizon_resolution_ = res;
	}

	void ScanRunCluster::setVerticalResolution(double res) {
		vertical_resolution_ = res;
	}

	void ScanRunCluster::setMinClusterSize(int min_cluster_size) {
		min_cluster_size_ = min_cluster_size;
	}

	void ScanRunCluster::setMaxClusterSize(int max_cluster_size) {
		max_cluster_size_ = max_cluster_size;
	}

	void ScanRunCluster::reorganized_cloud(pcl::PointCloud<PointT>::ConstPtr input_cloud) {

		for (int i = 0; i < input_cloud->points.size(); ++i) {
			auto &p = input_cloud->points[i];
			double dist_to_origin = sqrt(p.x * p.x + p.y * p.y);
			double angle = atan(p.z / dist_to_origin);
			double theta = atan2(p.y, p.x);
			int scan = calc_scan(scan_number_, angle);
			int idx = calc_col(horizon_resolution_, theta);
//			std::cout << "angle " << angle << " " << scan << " column " << theta << " " << idx << std::endl;
			if (scan >= this->scan_number_ || scan < 0) {
				continue;
			}
			if (idx >= col_size || idx < 0) {
				continue;
			}
//			assert(laser_frame_idx[scan][idx] == -1);
			laser_frame_idx[scan][idx] = i;
		}
	}

	void ScanRunCluster::segmentation(pcl::PointCloud<PointT>::ConstPtr input_cloud,
	                                  std::vector<pcl::PointIndicesPtr> &clusters) {
		col_size = calc_col(horizon_resolution_, PI);
		laser_frame_idx.resize(scan_number_);
		ROS_INFO("clear laser_frame_idx");
		for (int i = 0; i < scan_number_; ++i) {
			laser_frame_idx[i].resize(col_size, -1);
			for (int j = 0; j < col_size; ++j) {
				laser_frame_idx[i][j] = -1;
			}
		}
		laser_label.clear();
		laser_label.resize(input_cloud->points.size(), -1);

		LabelSet labelSet;
		reorganized_cloud(input_cloud);
		ROS_INFO("after reorganized_cloud scan %d, col %d", scan_number_, col_size);

		for (int j = 0; j < scan_number_; ++j) {
			find_runs_(input_cloud, j, labelSet);
		}
		ROS_INFO("after find Runs ...");
		for (int k = 0; k < scan_number_; ++k) {
//			update_labels_(input_cloud, k, label_set);
			update_labels_(input_cloud, k, labelSet);
		}
		ROS_INFO("after update labels");
		std::vector<std::pair<int, int> > sets;
		std::map<int, int> selected_sets;
//		label_set->getSets(sets);
		labelSet.getSets(sets);
		std::map<int, int> counter;
		for (int i = 0; i < input_cloud->points.size(); ++i) {
			if (laser_label[i] == -1) {
				continue;
			}
			int label = labelSet.findRoot(laser_label[i]);
			auto iter = counter.find(label);
			if (iter != counter.end()) {
				iter->second += 1;
			} else {
				counter.insert(std::make_pair(label, 1));
			}
		}
		int map_id = 0;
		for (auto iter = counter.begin(); iter != counter.end() ; ++iter) {
			if (iter->second >= min_cluster_size_ && iter->second <= max_cluster_size_) {
				selected_sets.insert(std::make_pair(iter->first, map_id ++));
			}
		}
		ROS_INFO("after map id");
		clusters.clear();
		for (int i = 0; i < map_id; ++ i) {
			pcl::PointIndicesPtr indices(new pcl::PointIndices);
			clusters.push_back(indices);
		}
		for (int i = 0; i < input_cloud->points.size(); ++i) {
			int label = laser_label[i];
			if (label == -1) {
				continue;
			}
			label = labelSet.findRoot(label);
			auto iter = selected_sets.find(label);
			if (iter != selected_sets.end()) {
				clusters[iter->second]->indices.push_back(i);
			}
		}
	}

	static double distance(const PointT &p, const PointT &q) {
		return sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z));
	}

	void ScanRunCluster::find_runs_(pcl::PointCloud<PointT>::ConstPtr input_cloud, int scan_line_, LabelSet& label_set) {
		auto &ring = laser_frame_idx[scan_line_];
		int begin = 0;
		while (begin < col_size && ring[begin ] == -1)
			begin += 1;
		int begin_idx = ring[begin];
		laser_label[begin_idx] = label_set.getNewLabel();
		for (int i = begin; i < col_size;) {
			int cur_idx = ring[i];
			int next = i + 1;
			while (ring[next] == -1 && next < col_size)
				next += 1;
			i = next;
			if (next == col_size) { //denote last point
				break;
			}
			int next_idx = ring[next];
			if (distance(input_cloud->points[cur_idx], input_cloud->points[next_idx]) <= row_dist_thresh_) {
				if (laser_label[next_idx] == -1) { // try to merge
					laser_label[next_idx] = laser_label[cur_idx];
					label_set.incrementToSet(laser_label[cur_idx], 1);
				} else {
					label_set.mergeLabel(laser_label[cur_idx], laser_label[next_idx]);
				}
			} else { // try to assign new label
				if (laser_label[next_idx] == -1) {
					laser_label[next_idx] = label_set.getNewLabel();
				}
			}
		}
		// for the begin and end
		int end = col_size - 1;
		while (end > begin && ring[end] == -1)
			end -= 1;
		if (end <= begin) {
			return;
		}
		int end_idx = ring[end];
		if (distance(input_cloud->points[begin_idx], input_cloud->points[end_idx]) < row_dist_thresh_) {
			if (laser_label[end_idx] == -1) {
				laser_label[end_idx] = laser_label[end_idx];
			} else {
				label_set.mergeLabel(laser_label[begin_idx], laser_label[end_idx]);
			}
		}
	}

//	void ScanRunCluster::update_labels_(pcl::PointCloud<PointT>::ConstPtr input_cloud, int scan_line_, LabelSet::Ptr label_set) {
void ScanRunCluster::update_labels_(pcl::PointCloud<pcl::PointXYZ>::ConstPtr input_cloud, int scan_line_,
                                    LidarSegmentation::LabelSet& label_set) {
		// search column
		auto & current_ring_idx = laser_frame_idx[scan_line_];
		for (int j = 0; j < col_size; ++ j) {
			if (current_ring_idx[j] == -1) {
				continue;
			}
			auto & p = input_cloud->points[current_ring_idx[j]];
			for (int i = scan_line_ - 1; i >= 0; -- i) {
				auto& prev_ring_idx = laser_frame_idx[i];
				if (prev_ring_idx[j] == -1) {
					continue;
				}
				auto &q = input_cloud->points[prev_ring_idx[j]];
				if (distance(p, q) < col_dist_thresh_) {
					label_set.mergeLabel(laser_label[prev_ring_idx[j]], laser_label[current_ring_idx[j]]);
				}
			}
		}
	}

	LabelSet::LabelSet() {
		forest.clear();
		forest.push_back(0);
		number.clear();
		number.push_back(0);
		max_label_ = 0;
	}

	void LabelSet::mergeLabel(int a, int b) {
		if (a > max_label_ || b > max_label_ || a <= 0 || b <= 0) {
			return;
		}
		int p_a = findRoot(a);
		int p_b = findRoot(b);
		forest[p_b] = p_a;
		number[p_a] += number[p_b];
	}

	int LabelSet::findRoot(int a) {
		if (a > max_label_ || a < 0) {
			return 0;
		}
		return forest[a] == a ? a : forest[a] = findRoot(forest[a]);
	}

	int LabelSet::getNewLabel() {
		forest.push_back(++ max_label_);
		number.push_back(1);
		assert(forest.size() == max_label_ + 1);
		assert(number.size() == max_label_ + 1);
		return max_label_;
	}

	int LabelSet::getElementsNumber(int a) {
		if (a > max_label_) {
			return 0;
		}
		int p = findRoot(a);
		return number[p];
	}

	void LabelSet::incrementToSet(int s, int n) {
		if (s > max_label_) {
			return ;
		}
		int p = findRoot(s);
		number[p] += n;
	}

	void LabelSet::getSets(std::vector<std::pair<int, int> > &sets) {
		sets.clear();
		for (int i = 1; i <= max_label_; ++i) {
			if (forest[i] == i) {
				std::pair<int, int> p = std::make_pair(i, number[i]);
				sets.push_back(p);
			}
		}
	}
}