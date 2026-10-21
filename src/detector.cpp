#include "cone_detection/detector.hpp"
#include "cone_detection/model_trt.hpp"
#ifdef WITH_ONNXRUNTIME
#include "cone_detection/model_onnx.hpp"
#endif

#include <algorithm>
#include <cctype>

namespace {
using Clock = std::chrono::steady_clock;

int64_t elapsedUs(const Clock::time_point &start, const Clock::time_point &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

rclcpp::Logger logger() {
    return rclcpp::get_logger("cone_detection");
}

std::string toLower(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
    return v;
}
}  // namespace

Precision precisionFromString(const std::string &value) {
    const std::string v = toLower(value);
    if (v == "fp32") return Precision::FP32;
    if (v == "fp16") return Precision::FP16;
    if (v == "int8") return Precision::INT8;
    throw std::invalid_argument("Unknown precision '" + value + "', expected fp32 | fp16 | int8");
}

std::string precisionToString(Precision precision) {
    switch (precision) {
        case Precision::FP32: return "fp32";
        case Precision::FP16: return "fp16";
        case Precision::INT8: return "int8";
    }
    return "unknown";
}

Backend backendFromString(const std::string &value) {
    const std::string v = toLower(value);
    if (v == "tensorrt" || v == "trt") return Backend::TensorRT;
    if (v == "onnx" || v == "onnxruntime") return Backend::OnnxRuntime;
    throw std::invalid_argument("Unknown backend '" + value + "', expected tensorrt | onnx");
}

std::string backendToString(Backend backend) {
    switch (backend) {
        case Backend::TensorRT: return "tensorrt";
        case Backend::OnnxRuntime: return "onnx";
    }
    return "unknown";
}

std::unique_ptr<Detector> makeDetector(const ModelParams &params) {
    switch (params.backend) {
        case Backend::TensorRT:
            return std::make_unique<TrtModel>(params);
        case Backend::OnnxRuntime:
#ifdef WITH_ONNXRUNTIME
            return std::make_unique<OnnxModel>(params);
#else
            throw std::runtime_error("backend 'onnx' requested, but cone_detection was built without "
                                     "ONNX Runtime (-DWITH_ONNXRUNTIME=ON)");
#endif
    }
    throw std::invalid_argument("Unknown backend");
}

Detector::Detector(const ModelParams &params) : m_config(params), numClasses(params.classes.size()) {}

void Detector::setInputSize(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::runtime_error("Invalid model input size " + std::to_string(height) + "x" + std::to_string(width));
    }
    imgH = height;
    imgW = width;
    const size_t planeSize = static_cast<size_t>(imgH) * imgW;
    // Contiguous planar buffer: split() writes each channel into its plane,
    // giving a CHW layout without an extra copy.
    gpu_dst = cv::cuda::GpuMat(1, static_cast<int>(planeSize), CV_8UC3);
    input_channels = {
        cv::cuda::GpuMat(imgH, imgW, CV_8U, &(gpu_dst.ptr()[0])),
        cv::cuda::GpuMat(imgH, imgW, CV_8U, &(gpu_dst.ptr()[planeSize])),
        cv::cuda::GpuMat(imgH, imgW, CV_8U, &(gpu_dst.ptr()[planeSize * 2]))
    };
    blob = cv::cuda::GpuMat(1, static_cast<int>(planeSize), CV_32FC3);
}

void Detector::setOutputShape(int32_t channels, int32_t anchors) {
    // YOLOv8 detect head: [1, 4 + num_classes, num_anchors]. A mismatch with
    // the classes list reads past the row in postprocessDetect().
    if (channels != static_cast<int32_t>(4 + numClasses) || anchors <= 0) {
        throw std::runtime_error("Model output is [1, " + std::to_string(channels) + ", " + std::to_string(anchors) +
                                 "], expected [1, 4 + " + std::to_string(numClasses) +
                                 " classes, N]. Check `classes` in params.yaml.");
    }
    numChannels = channels;
    numAnchors = anchors;
}

void Detector::warmUp(int iterations) {
    cv::cuda::GpuMat zeros(blob.size(), blob.type(), cv::Scalar::all(0));
    std::vector<float> scratch;
    for (int i = 0; i < iterations; ++i) {
        if (!infer(zeros, scratch)) {
            throw std::runtime_error("Warm-up inference failed");
        }
    }
}

void Detector::preprocess(const cv::cuda::GpuMat &gpuImg) {
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    // Scale from model input back to image coordinates
    m_ratio = 1.f / std::min(imgW / static_cast<float>(rgbMat.cols), imgH / static_cast<float>(rgbMat.rows));

    // Resize to the model input size, keep aspect ratio, pad right/bottom
    if (rgbMat.rows != imgH || rgbMat.cols != imgW) {
        rgbMat = resizeKeepAspectRatioPadRightBottom(rgbMat, imgH, imgW, cv::Scalar(0, 0, 0));
    }

    // NHWC uint8 -> NCHW float
    blobFromGpuMat(rgbMat);
}

void Detector::blobFromGpuMat(const cv::cuda::GpuMat &input) {
    cv::cuda::split(input, input_channels); // HWC -> CHW
    if (m_normalize) {
        // [0.f, 1.f]
        gpu_dst.convertTo(blob, CV_32FC3, 1.f / 255.f);
    } else {
        // [0.f, 255.f]
        gpu_dst.convertTo(blob, CV_32FC3);
    }

    // Mean subtraction and scaling. Skipped for the default values (0 and 1):
    // two full-image GPU passes per frame with no effect.
    const bool identity = m_subVals == std::array<float, 3>{0.f, 0.f, 0.f} &&
                          m_divVals == std::array<float, 3>{1.f, 1.f, 1.f};
    if (!identity) {
        cv::cuda::subtract(blob, cv::Scalar(m_subVals[0], m_subVals[1], m_subVals[2]), blob, cv::noArray(), -1);
        cv::cuda::divide(blob, cv::Scalar(m_divVals[0], m_divVals[1], m_divVals[2]), blob, 1, -1);
    }
}

void Detector::postprocessDetect(std::vector<float> &featureVector) {
    if (featureVector.size() < static_cast<size_t>(numChannels) * numAnchors) {
        RCLCPP_ERROR(logger(), "Output buffer smaller than expected, skipping postprocess");
        return;
    }
    output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        float* rowPtr = output.row(i).ptr<float>();
        float* bboxesPtr = rowPtr;
        float* scoresPtr = rowPtr + 4;
        float* maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > m_config.rect_confidence_threshold) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            int label = maxSPtr - scoresPtr;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, m_config.rect_confidence_threshold, m_config.iou_threshold, indices);
    ModelResult result{};
    for (int &chosenIdx : indices) {
        result.confidence = scores[chosenIdx];
        result.class_id = labels[chosenIdx];
        result.box = bboxes[chosenIdx];
        ret.push_back(result);
    }
    scores.clear();
    labels.clear();
    indices.clear();
    bboxes.clear();
}

std::vector<ModelResult> Detector::detect(const cv::Mat &inputImage) {
    const auto start = Clock::now();
    try {
        gpuImg.upload(inputImage);
    } catch (const cv::Exception &e) {
        RCLCPP_ERROR(logger(), "Error uploading image to GPU: %s", e.what());
        ret.clear();
        return ret;
    }
    const int64_t uploadUs = elapsedUs(start, Clock::now());
    detect(gpuImg);
    m_timings.upload_us = uploadUs;
    m_timings.total_us += uploadUs;
    return ret;
}

std::vector<ModelResult> Detector::detect(const cv::cuda::GpuMat &gpuImg) {
    m_timings = ModelTimings{};
    ret.clear();
    if (gpuImg.empty() || imgH == 0 || numAnchors == 0) {
        RCLCPP_ERROR(logger(), "detect: empty image or model not initialised");
        return ret;
    }

    // OpenCV CUDA calls in preprocess() run on the default stream and are
    // synchronous, so host timestamps measure the real GPU work per stage.
    // Both backends synchronise before infer() returns.
    const auto t0 = Clock::now();
    preprocess(gpuImg);
    const auto t1 = Clock::now();
    const bool succ = infer(blob, m_featureVector);
    const auto t2 = Clock::now();

    if (succ) {
        postprocessDetect(m_featureVector);
    } else {
        // Do not decode a stale or partially written output buffer
        RCLCPP_ERROR(logger(), "Error: Unable to run inference.");
    }
    const auto t3 = Clock::now();

    m_timings.preprocess_us = elapsedUs(t0, t1);
    m_timings.inference_us = elapsedUs(t1, t2);
    m_timings.postprocess_us = elapsedUs(t2, t3);
    m_timings.total_us = elapsedUs(t0, t3);
    return ret;
}

std::string Detector::get_class_by_id(int class_id) const {
    if (class_id < 0 || static_cast<size_t>(class_id) >= m_config.classes.size()) {
        RCLCPP_ERROR(logger(), "Invalid class_id: %d", class_id);
        return "unknown";
    }
    return m_config.classes[class_id];
}

cv::cuda::GpuMat Detector::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                               const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}
