//
// Created by yunle on 18-10-8.
//
#include <iostream>
#include "fast_ndt_slam/LidarMapping.h"
//#ifdef CUDA_FOUND
//#include <ndt_gpu/NormalDistributionsTransform.h>
//#endif
int main(int argc, char** argv) {
	std::cout<<(CUDA_FOUND? "CUDA_FOUND":"CUDA_NOT_FOUND")<<std::endl;
	ros::init(argc, argv, "fast_ndt_mapping");
	ros::NodeHandle nh;
	ros::NodeHandle private_nh("~");

	FAST_NDT::LidarMapping mapping;
	mapping.setup(nh, private_nh);
	ros::Rate(10);
	ros::spin();
	return 0;
}
