#include "cone_detection/cone_detection.hpp"

ConeDetection::ConeDetection(const rclcpp::NodeOptions &node_options) 
    : Node("cone_detection", node_options)
{
    mission_ = this->declare_parameter<std::string>("mission", "trackdrive");
    RCLCPP_INFO(
        this->get_logger(),
        "=== Started in mission mode: \033[1;32m%s\033[0m 🏁 ===",
        mission_.c_str()
    );
    

    lidar_points_topic_ = 
        this->declare_parameter<std::string>("lidar_points_topic");
    camera_image_topic_ = 
        this->declare_parameter<std::string>("camera_image_topic");
    
    // Synchronization of msgs from lidar_points_topic_ and camera_image_topic_
    
    // Define custom QoS profile for subscribers
    rmw_qos_profile_t lidar_camera_qos = rmw_qos_profile_default;
    lidar_camera_qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    lidar_camera_qos.depth = 2;
    lidar_camera_qos.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    lidar_camera_qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;

    // Define subscription options for subscribers
    rclcpp::SubscriptionOptions lidar_camera_options;
    lidar_camera_options.callback_group = nullptr;
    lidar_camera_options.use_default_callbacks = false;
    lidar_camera_options.ignore_local_publications = false;

    // Create subscribers with our custom profile and custom options
    lidar_points_subscriber_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, lidar_points_topic_, lidar_camera_qos, lidar_camera_options
    );
    camera_image_subscriber_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, camera_image_topic_, lidar_camera_qos, lidar_camera_options
    );

    // Create a synchronizer to handle messages from both subscribers
    lidar_camera_synchronizer_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>>>(
        message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image>(3),
        *lidar_points_subscriber_,
        *camera_image_subscriber_
    );

    // Register callback with the synchronizer
    lidar_camera_synchronizer_->registerCallback(std::bind(
        &ConeDetection::cone_detection_callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2
    ));


    ModelParams params;
    // Backend: "tensorrt" | "onnx" (ONNX Runtime, needs -DWITH_ONNXRUNTIME=ON).
    // Both use the same GPU preprocessing and postprocessing.
    params.backend = backendFromString(
        this->declare_parameter<std::string>("backend", "tensorrt"));
    // If using INT8 quantization, add calibration dataset to cone_detection/calibration_data
    // Precision: "fp32" | "fp16" | "int8". Empty -> legacy `use_fp16`
    // (true -> fp16, false -> int8).
    params.useFP16 = this->declare_parameter<bool>("use_fp16", false);
    const std::string precision =
        this->declare_parameter<std::string>("precision", "");
    params.precision = precision.empty()
        ? (params.useFP16 ? Precision::FP16 : Precision::INT8)
        : precisionFromString(precision);
    params.int8Fp16Fallback =
        this->declare_parameter<bool>("int8_fp16_fallback", false);
    params.calibrationBatchSize = static_cast<int32_t>(
        this->declare_parameter<int>("calibration_batch_size", 64));
    params.tagEngineWithPlatform =
        this->declare_parameter<bool>("tag_engine_with_platform", true);
    params.enginePath = this->declare_parameter<std::string>("engine_path");
    params.fp32EnginePath =
        this->declare_parameter<std::string>("fp32_engine_path", "");
    params.quantizedEnginePath = this->declare_parameter<std::string>("quantized_engine_path");
    params.onnxModelPath = this->declare_parameter<std::string>("model_path");
    params.calibrationDataPath = 
        this->declare_parameter<std::string>("calibration_data_path");
    params.onnxFp16ModelPath =
        this->declare_parameter<std::string>("onnx.fp16_model_path", "");
    params.onnxInt8ModelPath =
        this->declare_parameter<std::string>("onnx.int8_model_path", "");
    params.onnxCudaPluginPath =
        this->declare_parameter<std::string>("onnx.cuda_plugin_path", "");
    params.onnxCudnnConvAlgoSearch =
        this->declare_parameter<std::string>("onnx.cudnn_conv_algo_search", "EXHAUSTIVE");
    params.onnxIntraOpThreads = static_cast<int>(
        this->declare_parameter<int>("onnx.intra_op_num_threads", 1));
    params.onnxDisableCpuFallback =
        this->declare_parameter<bool>("onnx.disable_cpu_fallback", false);
    params.classes = 
        this->declare_parameter<std::vector<std::string>>("classes");
    params.img_size = {
        (int)this->declare_parameter<int>("width"),
        (int)this->declare_parameter<int>("height")
    };
    params.iou_threshold = this->declare_parameter<float>("iou_threshold");
    params.rect_confidence_threshold = 
        this->declare_parameter<float>("rect_confidence_threshold");

    model_ = makeDetector(params);
    init_benchmark();
    cone_height_ = this->declare_parameter<float>("cone_height");
    params_.max_len = this->declare_parameter<float>("max_len");
    params_.min_len = this->declare_parameter<float>("min_len");
    params_.interp_factor = this->declare_parameter<int>("interp_factor");

    if (params_.interp_factor <= 0) {
        RCLCPP_ERROR(
            this->get_logger(),
            "ConeDetection: interp_factor must be positive. Set to 1."
        );
        params_.interp_factor = 1;
    }

    if (params_.max_len <= 0.0f) {
        RCLCPP_ERROR(
            this->get_logger(),
            "ConeDetection: max_len must be positive. Set to 10 meters."
        );
        params_.max_len = 10.0f;
    }

    // Read and set matrices for lidar-camera fusion
    std::vector<double> temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("camera_matrix");
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> camera_matrix_d(
        temp_matrix_vec.data()
    );
    camera_matrix_ = camera_matrix_d.cast<float>();

    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("rotation_matrix");
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation_matrix_d(
        temp_matrix_vec.data()
    );
    rotation_matrix_ = rotation_matrix_d.cast<float>();

    temp_matrix_vec = 
        this->declare_parameter<std::vector<double>>("translation_matrix");
    Eigen::Map<Eigen::Matrix<double, 3, 1>> translation_matrix_d(
        temp_matrix_vec.data()
    );
    translation_matrix_ = translation_matrix_d.cast<float>();
    
    // Create transformation matrix
    transformation_matrix_ <<
        rotation_matrix_(0), rotation_matrix_(3), rotation_matrix_(6), translation_matrix_(0),
        rotation_matrix_(1), rotation_matrix_(4), rotation_matrix_(7), translation_matrix_(1),
        rotation_matrix_(2), rotation_matrix_(5), rotation_matrix_(8), translation_matrix_(2),
        0                  , 0                  , 0                  , 1;


    // Calculate height threshold in pixels for distance filtering
    // distance = height_real * focal_length / height_pixel
    // max_len is checked at node initialization!
    height_in_pixels = static_cast<int>(std::ceil(
        (cone_height_ * camera_matrix_.coeff(1, 1)) / params_.max_len
    ));


    detected_cones_publisher_ = 
        this->create_publisher<common_msgs::msg::ConeArray>("cone_detection/cone_array", 5);

    // Debug publishers
#ifndef NDEBUG
    detection_frames_publisher_ = 
        this->create_publisher<sensor_msgs::msg::Image>("cone_detection/image_detected", 5);
    cone_markers_publisher_ = 
        this->create_publisher<visualization_msgs::msg::MarkerArray>("cone_detection/cone_markers", 5);
    filtered_point_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/filtered_point_cloud", 5);
    interp_point_cloud_publisher_ = 
        this->create_publisher<sensor_msgs::msg::PointCloud2>("cone_detection/interp_point_cloud", 5);
#endif
}

void ConeDetection::cone_detection_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg
) {

    using BenchClock = std::chrono::steady_clock;
    const auto callback_start = BenchClock::now();
    FrameStats stats;

    #ifndef NDEBUG
        // Convert ROS image_msg to CV image
        cv_bridge::CvImagePtr cv_image_ptr;
        try {
            cv_image_ptr =
                cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Detect cones
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<std::string, cv::Rect>> detected_cones = 
            detect_cones_on_img(cv_image_ptr);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // Debug path: upload happens inside Model::detect(cv::Mat)
        stats.upload_us = model_->lastTimings().upload_us;
    #else

        // Upload the image to GPU memory directly from ROS message.
        // cuda_image_ is a member: reused between frames, so there is no
        // cudaMalloc per frame.
        cv::cuda::GpuMat &cuda_image = cuda_image_;
        const auto upload_start = std::chrono::high_resolution_clock::now();
        try {
            cv_bridge::CvImageConstPtr cv_image_const_ptr = 
                cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
            cuda_image.upload(cv_image_const_ptr->image);
        }
        catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "GPU upload failed: %s", e.what());
            return;
        }
        stats.upload_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - upload_start).count();


        // Detect cones
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<std::string, cv::Rect>> detected_cones = 
            detect_cones_on_img(cuda_image);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    #endif
    stats.detection_time_us = duration.count();
    stats.raw_cones = detected_cones.size();

    // Filter detected cones
    start = std::chrono::high_resolution_clock::now();
    detected_cones = filter_by_px_height(detected_cones);
    end = std::chrono::high_resolution_clock::now();
    stats.filter_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    stats.filtered_cones = detected_cones.size();

    start = std::chrono::high_resolution_clock::now();
    
    // Merge camera and lidar data and return closest points for each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> cone_positions = 
        lidar_camera_fusion(point_cloud_msg, image_msg, detected_cones);
    
    end = std::chrono::high_resolution_clock::now();
    duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.fusion_time_us = duration.count();
    stats.fused_candidates = cone_positions.size();

#ifndef NDEBUG
    int cone_marker_id_ = 0;
    visualization_msgs::msg::MarkerArray cone_markers_array_msg;
#endif
    // Creating message with detected cones for path planner
    common_msgs::msg::ConeArray cone_array_msg;

    for (const auto& pair : cone_positions) {
        pcl::PointXYZ cone_closest_point = pair.second;

        // Creating cone msg
        common_msgs::msg::Cone cone_msg;
        cone_msg.x = cone_closest_point.x;
        cone_msg.y = cone_closest_point.y;

        if (pair.first == "blue_cone") {
            cone_msg.side = Side::LEFT;
            cone_msg.color = Color::BLUE;
        } 
        else if (pair.first == "yellow_cone") {
            cone_msg.side = Side::RIGHT;
            cone_msg.color = Color::YELLOW;
        }
        else if (pair.first == "orange_cone") {
            cone_msg.color = Color::ORANGE;
        } 
        else if (pair.first == "large_orange_cone") {
            cone_msg.color = Color::LARGE_ORANGE;
        }
        else {
            continue; // Unknown cone
        }

        // Add a cone to the cone_array in the message
        cone_array_msg.cone_array.push_back(cone_msg);

#ifndef NDEBUG
        geometry_msgs::msg::Point point;
        point.x = cone_msg.x;
        point.y = cone_msg.y;
        point.z = -0.3;
        geometry_msgs::msg::Quaternion quaternion;
        quaternion.x = 0.0;
        quaternion.y = 0.0;
        quaternion.z = 0.0;
        quaternion.w = 1.0;
        geometry_msgs::msg::Pose pose;
        pose.position = point;
        pose.orientation = quaternion;
        geometry_msgs::msg::Vector3 vector3;
        vector3.x = 0.2;
        vector3.y = 0.2;
        vector3.z = 0.2;

        // Create cone marker message
        visualization_msgs::msg::Marker cone_marker_msg;
        cone_marker_msg.header.stamp = this->now();
        cone_marker_msg.header.frame_id = "/fsds/Lidar1";
        cone_marker_msg.ns = "basic_shapes";
        cone_marker_msg.id = cone_marker_id_++;
        cone_marker_msg.type = visualization_msgs::msg::Marker::CUBE;
        cone_marker_msg.action = visualization_msgs::msg::Marker::ADD;
        cone_marker_msg.pose = pose;
        cone_marker_msg.scale = vector3;
        // Set marker color
        if (cone_msg.color == Color::BLUE) {
            cone_marker_msg.color = VIZ_BLUE;
        } 
        else if (cone_msg.color == Color::YELLOW) {
            cone_marker_msg.color = VIZ_YELLOW;
        }
        else if (cone_msg.color == Color::ORANGE) {
            cone_marker_msg.color = VIZ_ORANGE;
        } 
        else if (cone_msg.color == Color::LARGE_ORANGE) {
            cone_marker_msg.color = VIZ_ORANGE;
        }
        else {
            cone_marker_msg.color = VIZ_GRAY;
        }
        cone_marker_msg.color.a = 1.0;
        cone_marker_msg.lifetime = 
            rclcpp::Duration(std::chrono::milliseconds(500));
        cone_marker_msg.frame_locked = false;
        // Push cone marker to cone markers array
        cone_markers_array_msg.markers.push_back(cone_marker_msg);
#endif
    }

    // Keep the input stamp: downstream nodes and the benchmark can then
    // compute sensor-to-output latency.
    cone_array_msg.header = image_msg->header;
    stats.published_cones = cone_array_msg.cone_array.size();
    detected_cones_publisher_->publish(cone_array_msg);
    stats.callback_total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        BenchClock::now() - callback_start).count();
    write_benchmark_row(image_msg->header, stats);
    write_detections(image_msg->header, last_results_);

#ifndef NDEBUG
    // Convert CV image to ROS msg and publish image message with
    // detected cones (labels and boxes)
    detection_frames_publisher_->publish(*cv_image_ptr->toImageMsg());
    
    cone_markers_publisher_->publish(cone_markers_array_msg);
#endif
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::detect_cones_on_img(
    cv_bridge::CvImagePtr cv_image_ptr
) {
    std::vector<std::pair<std::string, cv::Rect>> detected_cones;
    
    std::vector<ModelResult> results = model_->detect(cv_image_ptr->image);
    last_results_ = results;

    for (const auto& result : results) {
        detected_cones.push_back(std::make_pair(
            std::string(model_->get_class_by_id(result.class_id)),
            result.box
        ));
#ifndef NDEBUG
        cv::Scalar color(0, 256, 0);
        cv::rectangle(cv_image_ptr->image, result.box, color, 3);
        
        float confidence = floor(100 * result.confidence) / 100;
        std::string label = model_->get_class_by_id(result.class_id) + " " +
            std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

        cv::rectangle(
            cv_image_ptr->image,
            cv::Point(result.box.x, result.box.y - 25),
            cv::Point(result.box.x + label.length() * 10, result.box.y),
            color,
            cv::FILLED
        );

        cv::putText(
            cv_image_ptr->image,
            label,
            cv::Point(result.box.x, result.box.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            2
        );

        int center_x = result.box.x + result.box.width / 2;
        int center_y = result.box.y + result.box.height / 2;
        cv::circle(
            cv_image_ptr->image,
            cv::Point(center_x, center_y),
            5,
            color,
            -1
        );
#endif
    }

    return detected_cones;
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::detect_cones_on_img(
    const cv::cuda::GpuMat& cuda_image
) {
    std::vector<std::pair<std::string, cv::Rect>> detected_cones;
    
    std::vector<ModelResult> results = model_->detect(cuda_image);
    last_results_ = results;

    for (const auto& result : results) {
        detected_cones.push_back(std::make_pair(
            std::string(model_->get_class_by_id(result.class_id)),
            result.box
        ));
    }

    return detected_cones;
}

std::vector<std::pair<std::string, cv::Rect>> ConeDetection::filter_by_px_height(
    const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
) {
    std::vector<std::pair<std::string, cv::Rect>> filtered_cones;
    filtered_cones.reserve(detected_cones.size());

    for(const auto& cone : detected_cones) {
        if(cone.second.height >= height_in_pixels) {
            filtered_cones.push_back(cone);
        }
    }
    // copy ((
    return filtered_cones;
}

std::vector<std::pair<std::string, pcl::PointXYZ>> ConeDetection::lidar_camera_fusion(        
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &point_cloud_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &image_msg,
    const std::vector<std::pair<std::string, cv::Rect>> &detected_cones
) {
    // The closest point to each cone
    std::vector<std::pair<std::string, pcl::PointXYZ>> closest_points;

    // Convert ROS point_cloud_msg to PCL-compatible format
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    pcl::fromROSMsg(*point_cloud_msg, *point_cloud);

    if (point_cloud->empty()) {
        RCLCPP_ERROR(
            this->get_logger(),
            "lidar_camera_fusion: Input point cloud is empty!"
        );
        return closest_points;
    }

    // Filtering
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*point_cloud, *filtered_point_cloud, indices);
    filtered_point_cloud = distance_filter(filtered_point_cloud);
    filtered_point_cloud = ground_removal_filter(filtered_point_cloud);

    if (filtered_point_cloud->empty()) {
        RCLCPP_ERROR(
            this->get_logger(),
            "lidar_camera_fusion: Filtered point cloud is empty!"
        );
        return closest_points;
    }

    // Interpolation
    pcl::PointCloud<pcl::PointXYZ>::Ptr interpolated_point_cloud =
        interp_point_cloud(filtered_point_cloud);

    Eigen::Matrix<float, 4, 1> point_cloud_matrix;
    Eigen::Matrix<float, 3, 1> lidar_camera;
    int px, py;
    int img_width = static_cast<int>(image_msg->width);
    int img_height = static_cast<int>(image_msg->height);
    
    std::vector<ConeInfo> cone_infos;
    for (const auto& cone : detected_cones) {
        ConeInfo info;
        info.id = cone.first;
        info.bbox = cone.second;
        cone_infos.push_back(info);
    }

    // Set points for each cone bbox
    for (const auto& point : interpolated_point_cloud->points) {
        point_cloud_matrix << -point.y, -point.z, point.x, 1.0;

        lidar_camera = camera_matrix_ * (transformation_matrix_ * point_cloud_matrix);

        if (lidar_camera(2, 0) == 0.0f) {
            RCLCPP_ERROR(
                this->get_logger(),
                "lidar_camera_fusion: lidar_camera(2, 0) is zero!"
            );
            continue;
        }

        px = static_cast<int>(lidar_camera(0, 0) / lidar_camera(2, 0));
        py = static_cast<int>(lidar_camera(1, 0) / lidar_camera(2, 0));

        if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
            for (auto& cone : cone_infos) {
                if (cone.bbox.contains(cv::Point(px, py))) {
                    cone.associated_points.push_back(point);
                    break;
                }
            }
        }
    }

    // Remove overlay of bboxes
    for (size_t i = 0; i < cone_infos.size(); ++i) {
        for (size_t j = i + 1; j < cone_infos.size(); ++j) {
            cv::Rect intersection = cone_infos[i].bbox & cone_infos[j].bbox;
            if (intersection.area() > 0) {
                auto& points_i = cone_infos[i].associated_points;
                auto& points_j = cone_infos[j].associated_points;

                points_i.erase(
                    std::remove_if(points_i.begin(), points_i.end(), [&cone_infos, j](const pcl::PointXYZ& p) {
                        return cone_infos[j].bbox.contains(cv::Point(p.x, p.y));
                    }),
                    points_i.end()
                );

                points_j.erase(
                    std::remove_if(points_j.begin(), points_j.end(), [&cone_infos, i](const pcl::PointXYZ& p) {
                        return cone_infos[i].bbox.contains(cv::Point(p.x, p.y));
                    }),
                    points_j.end()
                );
            }
        }
    }

    // Select each closest point for each cone
    for (auto& cone : cone_infos) {
        if (cone.associated_points.empty()) continue;

        pcl::PointXYZ sum(0, 0, 0);
        float min_sqr_dist = std::numeric_limits<float>::max();
        pcl::PointXYZ closest_point;

        for (const auto& p : cone.associated_points) {
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;

            float sqr_dist = 
                std::pow(p.x - cone.average_point.x, 2) +
                std::pow(p.y - cone.average_point.y, 2) +
                std::pow(p.z - cone.average_point.z, 2);

            if (sqr_dist < min_sqr_dist) {
                min_sqr_dist = sqr_dist;
                closest_point = p;
            }
        }

        cone.average_point.x = sum.x / cone.associated_points.size();
        cone.average_point.y = sum.y / cone.associated_points.size();
        cone.average_point.z = sum.z / cone.associated_points.size();

        closest_points.push_back(std::make_pair(cone.id, closest_point));
    }
    // Remove cones that closer than 2 m of each other
    for (size_t i = 0; i < closest_points.size(); ++i) {
        for (size_t j = i + 1; j < closest_points.size(); ++j) {
            float sqr_dist = 
                std::pow(closest_points[i].second.x - closest_points[j].second.x, 2) +
                std::pow(closest_points[i].second.y - closest_points[j].second.y, 2) +
                std::pow(closest_points[i].second.z - closest_points[j].second.z, 2);
            
            // Comparing squared distances, 2.0 m
            if (sqr_dist < 4.0) {
                closest_points.erase(closest_points.begin() + j);
                --i;
                break;
            }
        }
    }

#ifndef NDEBUG
    sensor_msgs::msg::PointCloud2 temp_cloud;

    pcl::toROSMsg(*filtered_point_cloud, temp_cloud);
    temp_cloud.header = point_cloud_msg->header;
    filtered_point_cloud_publisher_->publish(temp_cloud);

    temp_cloud.data.clear();
    pcl::toROSMsg(*interpolated_point_cloud, temp_cloud);
    temp_cloud.header = point_cloud_msg->header;
    interp_point_cloud_publisher_->publish(temp_cloud);
#endif

    return closest_points;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::distance_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    // Reserve space
    filtered_point_cloud->points.reserve(point_cloud->points.size());

    // Calculate squared min and max lengths
    float min_len_squared = params_.min_len * params_.min_len;
    float max_len_squared = params_.max_len * params_.max_len;
    
    for (const auto& point : point_cloud->points) {
        // Calculate squared distance from origin
        float dist_squared = point.x * point.x + point.y * point.y;
        
        if (dist_squared < min_len_squared || dist_squared > max_len_squared)
            continue;
        
        filtered_point_cloud->points.push_back(point);
    }

    return filtered_point_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::ground_removal_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);
    // Coefficients of the plane model
    pcl::ModelCoefficients::Ptr model_coefficients(new pcl::ModelCoefficients);

    // Create the SAC segmentation object for ground plane extraction
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(false);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    // Max distance from the plane for a point to be considered part of it
    seg.setDistanceThreshold(0.05);

    seg.setInputCloud(point_cloud);
    // Apply segmentation to obtain the indices of the ground points (inliers)
    seg.segment(*inlier_indices, *model_coefficients);

    if (inlier_indices->indices.size() == 0) {
        PCL_ERROR("Ground plane not found.\n");
        return point_cloud;
    }

    // Extract the points that are not part of the ground (remove inliers)
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(point_cloud);
    extract.setIndices(inlier_indices);
    // True to remove the ground points
    extract.setNegative(true);
    extract.filter(*filtered_point_cloud);
    
    return filtered_point_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ConeDetection::interp_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr interpolated_point_cloud(
        new pcl::PointCloud<pcl::PointXYZ>
    );
    // Reserve space
    interpolated_point_cloud->points.reserve(
        // interp_factor is checked at node initialization!
        point_cloud->points.size() * params_.interp_factor
    );

    pcl::search::KdTree<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(point_cloud);
    
    int neighbors = 2;
    int found_neighbors;
    std::vector<int> k_indices;
    std::vector<float> k_sqr_distances;
    k_indices.reserve(neighbors);
    k_sqr_distances.reserve(neighbors);

    for (const auto& point : point_cloud->points) {
        found_neighbors = 
            kdtree.nearestKSearch(point, neighbors, k_indices, k_sqr_distances);
        
        if (found_neighbors < neighbors) continue;

        for (size_t k = 0; k < k_indices.size(); ++k) {
            const pcl::PointXYZ& neighbor =
                point_cloud->points[k_indices[k]];

            if (k_sqr_distances[k] < 0.1) {
                pcl::PointXYZ new_point(
                    (point.x + neighbor.x) / 2.0f,
                    (point.y + neighbor.y) / 2.0f,
                    (point.z + neighbor.z) / 2.0f
                );

                for (int i = 1; i < params_.interp_factor; ++i) {
                    pcl::PointXYZ interp_point;
                    // interp_factor is checked at node initialization!
                    interp_point.x = 
                        point.x + i * (new_point.x - point.x) / params_.interp_factor;
                    interp_point.y = 
                        point.y + i * (new_point.y - point.y) / params_.interp_factor;
                    interp_point.z = 
                        point.z + i * (new_point.z - point.z) / params_.interp_factor;
                    
                    interpolated_point_cloud->points.push_back(interp_point);
                }

                interpolated_point_cloud->points.push_back(new_point);
            }
        }
    }

    return interpolated_point_cloud;
}

void ConeDetection::init_benchmark() {
    const std::string csv_path =
        this->declare_parameter<std::string>("benchmark.csv_path", "");
    const std::string detections_path =
        this->declare_parameter<std::string>("benchmark.detections_csv_path", "");
    benchmark_warmup_frames_ =
        this->declare_parameter<int>("benchmark.warmup_frames", 20);
    const std::string label =
        this->declare_parameter<std::string>("benchmark.label", "");

    if (!csv_path.empty()) {
        benchmark_csv_.open(csv_path, std::ios::out | std::ios::trunc);
        if (!benchmark_csv_) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open benchmark CSV: %s", csv_path.c_str());
        } else {
            // Run metadata as comment lines; pandas: read_csv(comment='#')
            benchmark_csv_ << "# label=" << label << "\n"
                           << "# backend=" << model_->backend() << "\n"
                           << "# precision=" << precisionToString(model_->precision()) << "\n"
                           << "# engine=" << model_->modelPath() << "\n"
                           << "# runtime=" << model_->runtimeVersion() << "\n"
#ifdef NDEBUG
                           << "# build=release\n"
#else
                           << "# build=debug\n"
#endif
                           << "# warmup_frames=" << benchmark_warmup_frames_ << "\n";
            benchmark_csv_ << "frame,stamp_ns,upload_us,detection_time_us,model_total_us,"
                              "preprocess_us,inference_us,postprocess_us,filter_time_us,"
                              "fusion_time_us,callback_total_us,raw_cones,filtered_cones,"
                              "fused_candidates,published_cones\n";
            RCLCPP_INFO(this->get_logger(), "Benchmark CSV: %s", csv_path.c_str());
        }
    }
    if (!detections_path.empty()) {
        detections_csv_.open(detections_path, std::ios::out | std::ios::trunc);
        if (!detections_csv_) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open detections CSV: %s", detections_path.c_str());
        } else {
            detections_csv_ << "frame,stamp_ns,class_id,class_name,confidence,x,y,w,h\n";
        }
    }
}

void ConeDetection::write_benchmark_row(
    const std_msgs::msg::Header &image_header,
    const FrameStats &stats
) {
    const int64_t frame = frame_index_++;
    if (!benchmark_csv_.is_open() || frame < benchmark_warmup_frames_) {
        return;
    }
    const ModelTimings &t = model_->lastTimings();
    const int64_t stamp_ns = rclcpp::Time(image_header.stamp).nanoseconds();
    benchmark_csv_ << frame << ',' << stamp_ns << ','
                   << stats.upload_us << ',' << stats.detection_time_us << ','
                   << t.total_us << ',' << t.preprocess_us << ','
                   << t.inference_us << ',' << t.postprocess_us << ','
                   << stats.filter_time_us << ',' << stats.fusion_time_us << ','
                   << stats.callback_total_us << ','
                   << stats.raw_cones << ',' << stats.filtered_cones << ','
                   << stats.fused_candidates << ',' << stats.published_cones << '\n';
    // Flush every 100 rows: data survives Ctrl+C without a write per frame
    if (frame % 100 == 0) {
        benchmark_csv_.flush();
    }
}

void ConeDetection::write_detections(
    const std_msgs::msg::Header &image_header,
    const std::vector<ModelResult> &results
) {
    // frame_index_ was already incremented by write_benchmark_row()
    const int64_t frame = frame_index_ - 1;
    if (!detections_csv_.is_open() || frame < benchmark_warmup_frames_) {
        return;
    }
    const int64_t stamp_ns = rclcpp::Time(image_header.stamp).nanoseconds();
    for (const auto &r : results) {
        detections_csv_ << frame << ',' << stamp_ns << ',' << r.class_id << ','
                        << model_->get_class_by_id(r.class_id) << ',' << r.confidence << ','
                        << r.box.x << ',' << r.box.y << ',' << r.box.width << ','
                        << r.box.height << '\n';
    }
    if (frame % 100 == 0) {
        detections_csv_.flush();
    }
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConeDetection)
