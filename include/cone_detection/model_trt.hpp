#pragma once

#include "cone_detection/detector.hpp"

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};

/** TensorRT backend: builds (once) and runs a serialized engine. */
class TrtModel : public Detector {
public:
    explicit TrtModel(const ModelParams &config);
    ~TrtModel() override;

    std::string backend() const override { return "tensorrt"; }
    std::string modelPath() const override { return m_config.enginePath; }
    std::string runtimeVersion() const override;
    /** Engine file actually loaded (after platform tagging). */
    [[nodiscard]] const std::string &enginePath() const { return m_config.enginePath; }
    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const { return m_inputDims; };
    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const { return m_outputDims; };

protected:
    bool infer(const cv::cuda::GpuMat &blob, std::vector<float> &output) override;

private:
    bool buildEngine();
    /** INT8 build: pin the YOLOv8 detect-head post-processing to FP32. */
    void keepDetectHeadFp32(nvinfer1::INetworkDefinition &network, nvinfer1::IBuilderConfig &config);
    bool loadEngine();
    void clearGpuBuffers();

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    std::unique_ptr<nvinfer1::IInt8Calibrator> m_calibrator;

    Logger m_logger;
    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<size_t> m_outputElementSizes{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize = 1;
    cudaStream_t inferenceCudaStream = nullptr;
    std::vector<uint8_t> m_rawOutputBuffer;      // Scratch buffer for non-float output conversion
};
