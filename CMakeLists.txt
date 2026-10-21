cmake_minimum_required(VERSION 3.8)
project(cone_detection)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(USE_CUDA "Enable CUDA support" ON)
# ONNX Runtime backend (`backend: onnx`, src/model_onnx.cpp). ON by default
# when ONNXRUNTIME_ROOT is set in the environment (Dockerfile.jetpack sets
# /opt/onnxruntime on both architectures), OFF otherwise.
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(_with_ort_default ON)
else()
    set(_with_ort_default OFF)
endif()
option(WITH_ONNXRUNTIME "Build the ONNX Runtime backend" ${_with_ort_default})

# Default to an optimized build. Without it colcon builds without -O and
# without NDEBUG, which enables the debug drawing path and skews benchmarks.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(USE_CUDA)
    # CUDA 12.4+ (x86 legacy image) or 13.x (JetPack 7.2 image)
    find_package(CUDAToolkit 12.4 REQUIRED)
    add_definitions(-DUSE_CUDA)
else()
    set(USE_CUDA OFF)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
# find_package(pcl_ros REQUIRED)
# find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(common_msgs REQUIRED)
find_package(common REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# TensorRT from the NVIDIA apt repo installs to /usr/include/<arch> and
# /usr/lib/<arch>. CMake searches the <arch> subdirectories of /usr itself, so
# the same setting works for x86_64 and aarch64. Override with -DTensorRT_DIR=
# for a tarball install.
if(NOT TensorRT_DIR)
    set(TensorRT_DIR /usr)
endif()
add_subdirectory(libs/tensorrt-cpp-api)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

if(WITH_ONNXRUNTIME)
    # 1. -DONNXRUNTIME_ROOT=...  2. $ENV{ONNXRUNTIME_ROOT} (set in
    # Dockerfile.jetpack to /opt/onnxruntime)  3. legacy x86 image layout
    if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    endif()
    if(NOT ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_VERSION 1.18.1)
        if(USE_CUDA)
            set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime-libs/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
        else()
            set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime-libs/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
        endif()
    endif()
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    message(STATUS "ONNX Runtime backend: ${ONNXRUNTIME_LIB}")
endif()

include_directories(include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

ament_auto_find_build_dependencies()

set(CONE_DETECTION_SOURCES
    src/${PROJECT_NAME}.cpp
    src/detector.cpp
    src/model_trt.cpp
)
if(WITH_ONNXRUNTIME)
    list(APPEND CONE_DETECTION_SOURCES src/model_onnx.cpp)
endif()
ament_auto_add_library(${PROJECT_NAME} SHARED ${CONE_DETECTION_SOURCES})

rclcpp_components_register_node(
    ${PROJECT_NAME}
    PLUGIN "ConeDetection"
    EXECUTABLE ${PROJECT_NAME}_node
)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    tensorrt_cpp_api
)
# cudart comes through tensorrt_cpp_api (PUBLIC ${CUDA_LIBRARIES}). Do not
# add CUDA::cudart here: a second (shared) runtime next to the static one
# gives two independent CUDA runtime states in one process.
if(WITH_ONNXRUNTIME)
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_ONNXRUNTIME)
    # Find libonnxruntime.so (and the CUDA EP next to it) without LD_LIBRARY_PATH
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC libs/tensorrt-cpp-api/src)

# Offline detector benchmark (image folder -> same CSVs as the node)
ament_auto_add_executable(detector_bench src/detector_bench.cpp)
target_link_libraries(detector_bench ${PROJECT_NAME})
if(WITH_ONNXRUNTIME)
    set_target_properties(detector_bench PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib")
endif()


# tensorrt_cpp_api is a shared library built from the vendored subdirectory.
# Install it next to libcone_detection.so so the node finds it at runtime.
install(TARGETS tensorrt_cpp_api LIBRARY DESTINATION lib)

ament_auto_package(
    INSTALL_TO_SHARE
    config
    launch
)

install(FILES DESTINATION share/${PROJECT_NAME})
