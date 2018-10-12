cmake_minimum_required(VERSION 2.8.3)
project(cu_particle)

add_compile_options(-std=c+;11)

find_package(catkin REQUIRED
  roscpp
  std_msgs
)

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
  ${PROJECT_SOURCE_DIR}/cuPF_2/modules
  ${PROJECT_SOURCE_DIR}/cuPF_2/user
  ${PROJECT_SOURCE_DIR}/cuPF_2/modules/inc
  ${CATKIN_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

cuda_add_executable(cu_particle_node
  cuPF_2/main.cu
  cuPF_2/modules/cloud2data.cpp
  cuPF_2/modules/emicp.cu
  cuPF_2/modules/findRTfromS.cpp
  cuPF_2/particle_filter.cu
)

target_link_libraries(cu_particle_node
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
)
