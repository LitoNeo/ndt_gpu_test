cmake_minimum_required(VERSION 2.8.3)
project(ndt_mapping)

## Compile as C++11, supported in ROS Kinetic and newer
 add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PCL 1.7 REQUIRED)
find_package(CUDA)
if (CUDA_FOUND)
	add_definitions(-DCUDA_FOUND)
	include_directories(include)
    link_directories(lib)
	include_directories(${CUDA_INCLUDE_DIRS})
endif ()

# include_directories(include)
# link_directories(include)

# include_directories(lib)
# link_directories(lib)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
#  ndt_gpu
#	ndt_cpu
  pcl_conversions
#  pcl_omp_registration
  roscpp
  sensor_msgs
		tf
  std_msgs
		geometry_msgs
		common_msgs
#  velodyne_pointcloud
)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   nav_msgs#   sensor_msgs#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fast_ndt_slam
  CATKIN_DEPENDS nav_msgs roscpp tf sensor_msgs std_msgs geometry_msgs pcl_conversions common_msgs
# ndt_cpu pcl_omp_registration velodyne_pointcloud
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include ${catkin_INCLUDE_DIRS}
  include/
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/fast_ndt_slam.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure


#set(SRC_LIST include/fast_ndt_slam/LidarMapping.h src/LidarMapping.cpp src/Cluster.cpp src/ClusterDetector.cpp src/FastClusterDetector.cpp)
set(SRC_LIST include/fast_ndt_slam/LidarMapping.h src/LidarMapping.cpp)

## Declare a C++ executable
add_executable(fast_ndt_slam_node src/fast_ndt_slam_node.cpp ${SRC_LIST})
if (CUDA_FOUND)
    message("link ndt_cpu ndt_gpu to target------------------------------>>>>>>>")
	target_link_libraries(fast_ndt_slam_node ndt_gpu ndt_cpu) #这里的align改成你的app
endif ()

#add_executable(ScanLineRun src/cluster_node.cpp src/ScanLineRun.cpp include/detect/ScanLineRun.h)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against

target_link_libraries(fast_ndt_slam_node ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_LIBS})





#target_link_libraries(ScanLineRun ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
