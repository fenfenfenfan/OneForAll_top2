PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)
# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

set(ENABLE_LITE_BACKEND OFF)
#set(FDLIB ${FASTDEPLOY_INSTALL_DIR})

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})
include_directories(${FastDeploy_INCLUDE_DIRS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# 添加FastDeploy库依赖
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
