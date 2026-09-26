cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  /usr/include/eigen3
  /opt/unitree_robotics/include/ddscxx
  /opt/unitree_robotics/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_directories(
  /opt/unitree_robotics/lib
  /usr/local/lib
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
)

file(GLOB_RECURSE ADD_SRC_LIST src/*.cpp)

# =========================================================
# 1. MODO CIEGO (go2_ctrl)
# =========================================================
add_library(go2_lib ${ADD_SRC_LIST})

add_executable(go2_ctrl main.cpp)
target_link_libraries(go2_ctrl go2_lib)

# =========================================================
# 2. MODO VISIÓN (go2lidar_ctrl)
# =========================================================
add_library(go2lidar_lib ${ADD_SRC_LIST})
target_compile_definitions(go2lidar_lib PRIVATE USE_ROS2_VISION)
ament_target_dependencies(go2lidar_lib rclcpp std_msgs)

add_executable(go2lidar_ctrl main.cpp)
target_compile_definitions(go2lidar_ctrl PRIVATE USE_ROS2_VISION)
target_link_libraries(go2lidar_ctrl go2lidar_lib)
ament_target_dependencies(go2lidar_ctrl rclcpp std_msgs)
