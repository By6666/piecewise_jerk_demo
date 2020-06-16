# 最小版本要求
cmake_minimum_required(VERSION 3.4.1)

# 项目名称 区分大小写
project(PRINT)

set(OUTPUT_NAME "exe.out")

# 设置编译器
set(CMAKE_CXX_COMPILER "g++")

# 设置编译选项
add_definitions(-std=c++11)

include_directories(${PROJECT_SOURCE_DIR}/piecewise_jerk)
include_directories(${PROJECT_SOURCE_DIR}/matplot)

# 获得所有source文件
aux_source_directory(${PROJECT_SOURCE_DIR}/piecewise_jerk SRC_LIST)


message(STATUS "PYTHONHOME" ${PYTHONHOME})
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

# 寻找package
find_package(osqp REQUIRED)
find_package(glog REQUIRED)
find_package(PythonLibs 2.7)


add_executable(${OUTPUT_NAME} ${SRC_LIST} main.cpp)
target_include_directories(${OUTPUT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${OUTPUT_NAME} PRIVATE osqp::osqp glog ${PYTHON_LIBRARIES})
