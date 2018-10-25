cmake_minimum_required(VERSION 2.8.3)
project(cu_particle)
set(CMAKE_VERBOSE_MAKEFILE FALSE)


find_package(
  catkin REQUIRED
  roscpp
  std_msgs
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCL REQUIRED pcl_common-1.8 pcl_io-1.8 pcl_visualization-1.8 pcl_sample_consensus-1.8)
pkg_check_modules(EIGEN3 REQUIRED eigen3)


# Boost
find_package(Boost 1.54 REQUIRED)

# Atlas
find_library(ATLAS atlas)

# Lapack
find_package(LAPACK REQUIRED)
add_definitions ( ${LAPACK_LINKER_FLAGS} )


option(ENABLE_CUDA "Enable the cuda build" ON)
if(ENABLE_CUDA)
  find_package(CUDA 9.0 REQUIRED)
endif()

catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
)


include_directories(
  ${PROJECT_SOURCE_DIR}/cuPF_2
  ${PROJECT_SOURCE_DIR}/cuPF_2/modules/inc
  ${PROJECT_SOURCE_DIR}/cub
  ${CATKIN_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  /opt/ros/kinetic/include
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60 -O3 --use_fast_math -std=c++11")
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

#???
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

add_definitions(-D_GLIBCXX_USE_C99_MATH=1)

cuda_add_executable(cu_particle_node
  cuPF_2/main.cpp
  cuPF_2/particle_filter.cu
  cuPF_2/modules/emicp.cu
  cuPF_2/modules/findRTfromS.cpp
  cuPF_2/devices/lidar.cu
  cuPF_2/devices/odometry.cu
  cuPF_2/devices/map.cu
  
)

cuda_add_cublas_to_target(cu_particle_node)

target_link_libraries(cu_particle_node
 ${catkin_LIBRARIES}
 ${CUDA_LIBRARIES}
 ${CUDA_curand_LIBRARY}
 ${PCL_LIBRARIES}
 ${LAPACK_LIBRARIES}
)
