cmake_minimum_required(VERSION 3.10)

# == Project ==
project(orbslam2_bridge)

# == Check C++14 ==
include(CheckCXXCompilerFlag)
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++14" COMPILER_SUPPORTS_CXX14)
if(NOT ${COMPILER_SUPPORTS_CXX14})
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} doesn't support C++14\n")
endif()

# == Use C++14 ==
set(CMAKE_CXX_STANDARD 14)
message("Compiler:\n\t${CMAKE_CXX_COMPILER} (using C++14)")

# == Set default build type to release ==
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
endif()
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
message("Build Type:\n\t${CMAKE_BUILD_TYPE}")

# == Clear "CMAKE_CXX_FLAGS" ==
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pipe -fopenmp -Ofast -lstdc++fs -mfpmath=both -mtune=native"
)# -mtune=native

# == Set warning flags ==
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas)
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()
message("CMAKE_CXX_FLAGS:\n\t${CMAKE_CXX_FLAGS}")

# == Point Cloud Library ==
find_package(PCL QUIET REQUIRED COMPONENTS common io registration visualization kdtree)
include_directories(SYSTEM ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
message(STATUS "PCL version:\n\t${PCL_VERSION}")

# == OpenCV ==
find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version:\n\t${OpenCV_VERSION}")

# == Eigen3 ==
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 version:\n\t${EIGEN3_VERSION_STRING}")

# == Catkin ==
find_package(catkin REQUIRED COMPONENTS cv_bridge image_transport tf)
catkin_package(CATKIN_DEPENDS cv_bridge image_transport tf)

# == orbslam ==
execute_process(COMMAND ./build.sh
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/3rd/ORB_SLAM2")
add_subdirectory(3rd/ORB_SLAM2)

# == Headers and sources ==
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
# cmake-format: off
include_directories(
  SYSTEM ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}/3rd
  ${CMAKE_CURRENT_LIST_DIR}/3rd/ORB_SLAM2 )
# cmake-format: on

# == Executable ==
add_executable(orbslam2_bridge_node src/bridge_node.cpp)
target_link_libraries(orbslam2_bridge_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ORB_SLAM2)

add_executable(orbslam2_stereo_bridge_node src/stereo_bridge_node.cpp)
target_link_libraries(orbslam2_stereo_bridge_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ORB_SLAM2)
