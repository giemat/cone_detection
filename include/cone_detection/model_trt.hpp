#pragma once

#include "cone_detection/detector.hpp"

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_fp16.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override;
};

// Class used for int8 calibration
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(int32_t batchSize, int32_t inputW, int32_t inputH, const std::string &calibDataDirPath,
                           const std::string &calibTableName, const std::string &inputBlobName,
                           const std::array<float, 3> &subVals, const std::array<float, 3> &divVals,
                           bool normalize, bool readCache = true);
    virtual ~Int8EntropyCalibrator2();
    // Abstract base class methods which must be implemented
    int32_t getBatchSize() const noexcept override;
    bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept override;
    void const *readCalibrationCache(std::size_t &length) noexcept override;
    void writeCalibrationCache(void const *ptr, std::size_t length) noexcept override;

private:
    const int32_t m_batchSize;
    const int32_t m_inputW;
    const int32_t m_inputH;
    int32_t m_imgIdx;
    std::vector<std::string> m_imgPaths;
    size_t m_inputCount;
    const std::string m_calibTableName;
    const std::string m_inputBlobName;
    const std::array<float, 3> m_subVals;
    const std::array<float, 3> m_divVals;
    const bool m_normalize;
    const bool m_readCache;
    void *m_deviceInput;
    std::vector<char> m_calibCache;
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
    bool loadEngine();
    void clearGpuBuffers();

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    std::unique_ptr<Int8EntropyCalibrator2> m_calibrator;

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
