#ifndef CONE_DETECTION_HPP
#define CONE_DETECTION_HPP

// Show debug data in release and debug modes
//#undef NDEBUG

#include <rclcpp/rclcpp.hpp>
// Matrix manipulations
#include <Eigen/Dense>
// Image manipulations, msg<->Image conversion
#include <opencv2/opencv.hpp>
// cv_bridge.h is deprecated (Humble) and removed in newer distros
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
// Point cloud manipulations
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
// Synchronization of messages from camera and lidar
#include <rclcpp/qos.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
// Messages
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "common_msgs/msg/cone_array.hpp"
#include "common_msgs/msg/cone.hpp"
// Image detection using a model (TensorRT or ONNX Runtime backend)
#include "detector.hpp"
//From common
#include "common/cone.hpp"
#include "common/viz.hpp"
// Standard
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
// Debug
#ifndef NDEBUG
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#endif

/**
 * Per-frame benchmark record. One CSV row per processed frame, same column
 * names as the cone_detection_summary.csv produced earlier on x86.
 */
struct FrameStats {
    int64_t upload_us = 0;
    int64_t detection_time_us = 0;
    int64_t filter_time_us = 0;
    int64_t fusion_time_us = 0;
    int64_t callback_total_us = 0;
    size_t raw_cones = 0;
    size_t filtered_cones = 0;
    size_t fused_candidates = 0;
    size_t published_cones = 0;
};

/**  */
struct ConeInfo {
    std::string id;
    cv::Rect bbox;
    std::vector<pcl::PointXYZ> associated_points;
    pcl::PointXYZ average_point;
    float confidence;
};

/**
 * Lidar-Camera fusion parameters. All parameters can be changed
 * in `params.yaml`.
 * @param max_len Maximum distance for the point cloud. Points farther than
 * this distance are filtered out.
 * @param min_len Minimum distance for the point cloud. Points closer than
 * this distance are filtered out.
 * @param interp_factor Interpolation factor.
 */
typedef struct {
    float max_len;
    float min_len;
    int interp_factor;
} FusionParams;

/**
 * The ROS2 node responsible for detecting the cones using the lidar and
 * camera and sending the position of the cones relative to the lidar.
 */
class ConeDetection : public rclcpp::Node {
public:
    /**
     * Constructor. Reads parameters from config, synchronizes lidar
     * and camera topics, creates subscribers and publishers.
     */
    ConeDetection(const rclcpp::NodeOptions &node_options);
private:
    float cone_height_;
    /** Lidar point cloud topic name. Can be changed in `params.yaml`. */
    std::string lidar_points_topic_;
    /** Сamera image topic тame. Can be changed in `params.yaml`. */
    std::string camera_image_topic_;
    /** Subscriber on lidar_points_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> lidar_points_subscriber_;
    /** Subscriber on camera_image_topic_. */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> camera_image_subscriber_;
    /**
     * Synchronizer that synchronizes messages from the lidar and camera
     * topics using an approximate time-based policy.
     */
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>> lidar_camera_synchronizer_;

    /** Publisher for detected cones. */
    rclcpp::Publisher<common_msgs::msg::ConeArray>::SharedPtr detected_cones_publisher_;
// CMake macro for debug build
#ifndef NDEBUG
    /** Publishes image with labeled and boxed detected cones. */
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr detection_frames_publisher_;
    /** Publishes markers for visualization. */
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cone_markers_publisher_;
    /** Publishes filtered point cloud. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_point_cloud_publisher_;
    /** Publishes interpolated point cloud. */
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr interp_point_cloud_publisher_;
#endif

    /**
     * Main callback of cone detection. Receives messages from camera and
     * lidar, detects cones and merges data from these two sensors,
     * sending at the end array of detected cones `ConeArray`.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     */
    void cone_detection_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg
    );

    /** Detector (backend selected by the `backend` parameter). */
    std::unique_ptr<Detector> model_;
    /** Camera frame on the GPU (release path), reused between frames. */
    cv::cuda::GpuMat cuda_image_;

    /**
     * Detects cones using a model.
     * @param cv_image_ptr image pointer from `cv_bridge`, received by
     * converting an `Image` message to `OpenCV` readable format.
     * @return vector of pairs of cone class name and cone box (`cv::Rect`).
     */
    std::vector<std::pair<std::string, cv::Rect>> detect_cones_on_img(
        cv_bridge::CvImagePtr cv_image_ptr
    );
    /**
     * Detects cones using a model.
     * @param cuda_image image in `cv::cuda::GpuMat` format.
     * @return vector of pairs of cone class name and cone box
     * (`cv::Rect`).
     */
    std::vector<std::pair<std::string, cv::Rect>> detect_cones_on_img(
        const cv::cuda::GpuMat& cuda_image
    );

    /** Height threshold in pixels for distance filtering. */
    int height_in_pixels;

    /**
     * Filters cones by `cv::Rect` box height in pixels.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) for filtering.
     * @return filtered vector of pairs.
     */
    std::vector<std::pair<std::string, cv::Rect>> filter_by_px_height(
        const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
    );

    /** Lidar-camera fusion parameters. */
    FusionParams params_;
    /** Сamera's intrinsic parameters matrix; camera calibration matrix. */
    Eigen::Matrix<float, 3, 4> camera_matrix_;
    /** Describes the rotation of the camera regarding the lidar sensor. */
    Eigen::Matrix<float, 3, 3> rotation_matrix_;
    /**
     * Represents the translation from lidar coordinates to camera
     * coordinates.
     */
    Eigen::Matrix<float, 3, 1> translation_matrix_;
        /** 
     * Lidar-camera transformation (rotation and translation) matrix;
     * extrinsic parameters between the sensors.
     */
    Eigen::Matrix<float, 4, 4> transformation_matrix_;

    /**
     * Finds the closest point of the cone relative to the lidar.
     * @param point_cloud_msg `PointCloud2` from lidar.
     * @param image_msg `Image` from camera.
     * @param detected_cones vector of pairs of cone class name and cone
     * box (`cv::Rect`) received from `detect_cones_on_img()`.
     * @return vector of pairs of cone class name and cone closest point
     * (`pcl::PointXYZ`).
     */
    std::vector<std::pair<std::string, pcl::PointXYZ>> lidar_camera_fusion(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
        const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
    );

    /**
     * Filters point cloud based on min and max distances from config. 
     * @param point_cloud point cloud for filtering.
     * @return filtered point cloud.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr distance_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /**
     * Removes ground from point cloud using SAC segmentation.
     * @param point_cloud point cloud for ground removing.
     * @return point cloud without ground.
     * @todo move distance threshold to config.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_removal_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /**
     * Interpolates point cloud. Uses KdTree to perform nearest-neighbor
     * searches for interpolation between points in point cloud.
     * @param point_cloud point cloud for interpolation.
     * @return interpolated point cloud.
     * @todo move neighbors number to config.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr interp_point_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
    );

    /** The current dynamic event (mission). */
    std::string mission_;

    /** Benchmark output. Disabled when `benchmark.csv_path` is empty. */
    std::ofstream benchmark_csv_;
    /** Detections per frame (class, confidence, box) for accuracy comparison. */
    std::ofstream detections_csv_;
    /** Frames skipped before logging (engine warm-up, lazy CUDA init). */
    int64_t benchmark_warmup_frames_ = 0;
    int64_t frame_index_ = 0;
    /** Open the benchmark CSV files and write the run header. */
    void init_benchmark();
    void write_benchmark_row(
        const std_msgs::msg::Header &image_header,
        const FrameStats &stats
    );
    void write_detections(
        const std_msgs::msg::Header &image_header,
        const std::vector<ModelResult> &results
    );
    /** Raw model output of the current frame, kept for detections_csv_. */
    std::vector<ModelResult> last_results_;
};

#endif
