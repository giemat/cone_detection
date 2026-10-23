#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Utility methods
namespace ConeUtil {
    inline bool doesFileExist(const std::string &filepath) {
        std::ifstream f(filepath.c_str());
        return f.good();
    }

    inline void checkCudaErrorCode(cudaError_t code) {
        if (code != 0) {
            std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                                 "), with message: " + cudaGetErrorString(code);
            std::cout << errMsg << std::endl;
            throw std::runtime_error(errMsg);
        }
    }
}

typedef struct {
    int class_id;
    float confidence;
    cv::Rect box;
} ModelResult;

/** Model precision. */
enum class Precision { FP32, FP16, INT8 };

/** Parses "fp32" | "fp16" | "int8" (case-insensitive). Throws on other values. */
Precision precisionFromString(const std::string &value);
std::string precisionToString(Precision precision);

/** Inference backend. Pre- and postprocessing are the same for both. */
enum class Backend { TensorRT, OnnxRuntime };

/** Parses "tensorrt" | "onnx" (case-insensitive). Throws on other values. */
Backend backendFromString(const std::string &value);
std::string backendToString(Backend backend);

typedef struct {
    Backend backend = Backend::TensorRT;
    Precision precision = Precision::INT8;
    /** FP32 ONNX model. TensorRT builds its engines from it; ONNX Runtime runs it for fp32. */
    std::string onnxModelPath;
    std::vector<std::string> classes;
    std::vector<int> img_size = { 1280, 1280 };
    float iou_threshold = 0.5;
    float rect_confidence_threshold = 0.6;

    // TensorRT
    /** FP16 engine path (legacy name kept for params.yaml compatibility). */
    std::string enginePath;
    std::string fp32EnginePath;
    std::string quantizedEnginePath;
    std::string calibrationDataPath;
    /** INT8 calibrator: "minmax" (IInt8MinMaxCalibrator) or "entropy" (IInt8EntropyCalibrator2). */
    std::string calibrationAlgorithm = "minmax";
    /** Calibration images, evenly spread over the sorted folder (0 = all). Same rule as export_onnx_variants.py. */
    int calibrationMaxImages = 1000;
    /** Legacy flag: true -> FP16, false -> INT8. Ignored when precision is set. */
    bool useFP16 = false;
    /** INT8 build: also allow FP16 kernels for layers without INT8 support. */
    bool int8Fp16Fallback = false;
    /**
     * INT8 build: keep the YOLOv8 detect-head post-processing (DFL, box
     * decode, sigmoid, output concat) in FP32. Without it the shared INT8
     * scale of boxes (pixels) and scores (0..1) zeroes the scores.
     */
    bool int8KeepHeadFp32 = true;
    /**
     * Append "_sm<cc>_trt<version>" to engine file names. Engines are only
     * valid for the GPU + TensorRT version that built them, so this keeps x86
     * and Jetson engines apart when ws/ is shared or copied between machines.
     */
    bool tagEngineWithPlatform = true;

    // ONNX Runtime
    /** FP16 / INT8 (QDQ) ONNX files, see scripts/benchmark/export_onnx_variants.py. */
    std::string onnxFp16ModelPath;
    std::string onnxInt8ModelPath;
    /**
     * CUDA plugin execution provider library. Empty: classic CUDA EP linked
     * into ONNX Runtime (x86 GPU build). Set on aarch64, where ONNX Runtime
     * is a CPU build and CUDA comes as a separate plugin.
     */
    std::string onnxCudaPluginPath;
    /** cuDNN convolution algorithm search: EXHAUSTIVE | HEURISTIC | DEFAULT. */
    std::string onnxCudnnConvAlgoSearch = "EXHAUSTIVE";
    int onnxIntraOpThreads = 1;
    /**
     * Fail at startup if any node would run on the CPU EP. Proves that the
     * whole graph runs on the GPU; QDQ (int8) models may need it off.
     */
    bool onnxDisableCpuFallback = false;
} ModelParams;

/** Per-frame timings of the last Detector::detect() call, in microseconds. */
typedef struct {
    int64_t upload_us = 0;
    int64_t preprocess_us = 0;
    int64_t inference_us = 0;
    int64_t postprocess_us = 0;
    int64_t total_us = 0;
} ModelTimings;

/**
 * YOLOv8 detector. Owns the GPU preprocessing (BGR->RGB, letterbox,
 * HWC->CHW, /255) and the postprocessing (decode, NMS), so every backend
 * runs exactly the same code around the network. A backend only implements
 * infer(): device input blob in, host float output out.
 *
 * Timing stages (ModelTimings):
 *   preprocess  colour conversion, letterbox resize, CHW float blob (GPU)
 *   inference   backend run, including the device->host output copy
 *   postprocess decode + NMS (CPU)
 */
class Detector {
public:
    explicit Detector(const ModelParams &params);
    virtual ~Detector() = default;
    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    std::vector<ModelResult> detect(const cv::Mat &inputImage);
    std::vector<ModelResult> detect(const cv::cuda::GpuMat &gpuImg);
    std::string get_class_by_id(int class_id) const;

    /** Timings of the last detect() call. */
    [[nodiscard]] const ModelTimings &lastTimings() const { return m_timings; }
    [[nodiscard]] Precision precision() const { return m_config.precision; }
    /** "tensorrt" | "onnx" */
    [[nodiscard]] virtual std::string backend() const = 0;
    /** Engine / model file actually loaded. */
    [[nodiscard]] virtual std::string modelPath() const = 0;
    /** Backend library version for the CSV header, e.g. "tensorrt=101601". */
    [[nodiscard]] virtual std::string runtimeVersion() const = 0;

protected:
    /**
     * Runs the network on the CHW float blob in device memory
     * (1 x 3 x imgH x imgW) and writes the first output, converted to float,
     * to `output` (numChannels x numAnchors). Returns false on failure.
     */
    virtual bool infer(const cv::cuda::GpuMat &blob, std::vector<float> &output) = 0;

    /** Call once the backend knows its input size (H, W). */
    void setInputSize(int height, int width);
    /** Call once the backend knows its output shape [1, channels, anchors]. Throws on a class count mismatch. */
    void setOutputShape(int32_t channels, int32_t anchors);
    /**
     * Runs infer() on a zero blob. Call at the end of the backend
     * constructor, so lazy initialisation (cuDNN algorithm search, kernel
     * loading) happens before the first camera frame.
     */
    void warmUp(int iterations = 3);

    ModelParams m_config;
    // Normalisation of the blob. The TensorRT INT8 calibrator uses the same values.
    bool m_normalize = true;
    std::array<float, 3> m_subVals{0.f, 0.f, 0.f};
    std::array<float, 3> m_divVals{1.f, 1.f, 1.f};

private:
    void preprocess(const cv::cuda::GpuMat &gpuImg);
    void blobFromGpuMat(const cv::cuda::GpuMat &input);
    void postprocessDetect(std::vector<float> &featureVector);

    ModelTimings m_timings;

    // Model input size and planar buffers matching it (setInputSize)
    int imgH = 0;
    int imgW = 0;
    cv::cuda::GpuMat gpu_dst;
    std::vector<cv::cuda::GpuMat> input_channels;

    float m_imgWidth = 0;
    float m_imgHeight = 0;
    float m_ratio = 1.0f;

    // Preallocated data structures for performance
    cv::cuda::GpuMat rgbMat;                     // For color conversion
    cv::cuda::GpuMat gpuImg;                     // For input image upload
    cv::cuda::GpuMat blob;                       // CHW float network input
    cv::Mat output;                              // For postprocessing output
    std::vector<ModelResult> ret;                // Detection results
    std::vector<float> m_featureVector;          // Inference output

    std::vector<cv::Rect> bboxes;
    cv::Rect_<float> bbox;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    int32_t numChannels = 0;
    int32_t numAnchors = 0;
    size_t numClasses = 0;
};

/**
 * Calibration image list: .jpg/.jpeg/.png/.bmp files directly in `dir`,
 * sorted by path, then `maxImages` of them evenly spread (index
 * floor(i * N / maxImages)); 0 = all. scripts/benchmark/export_onnx_variants.py
 * uses the same rule, so TensorRT and ONNX Runtime INT8 see the same images.
 */
std::vector<std::string> calibrationImages(const std::string &dir, size_t maxImages);

/** Letterbox resize to height x width, keep aspect ratio, pad right/bottom. */
cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                     const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0));

/**
 * BGR image -> 1x3xHxW float blob (RGB, letterbox, /255) in `blob`
 * (1 x H*W, CV_32FC3, planar). Same steps as Detector::preprocess(); used by
 * the INT8 calibrator so calibration sees exactly the inference input.
 */
void makeInputBlob(const cv::cuda::GpuMat &bgr, int height, int width, cv::cuda::GpuMat &blob);

/** Creates the backend selected by params.backend. */
std::unique_ptr<Detector> makeDetector(const ModelParams &params);
