#include "cone_detection/model_trt.hpp"

#include <fstream>

namespace {
rclcpp::Logger logger() {
    return rclcpp::get_logger("cone_detection");
}

/** "dir/model.engine" + "_x" -> "dir/model_x.engine" */
std::string insertBeforeExtension(const std::string &path, const std::string &tag) {
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of('/');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path + tag;
    }
    return path.substr(0, dot) + tag + path.substr(dot);
}

/** "model.engine" -> "model_sm110_trt101601.engine" */
std::string tagEnginePath(const std::string &path) {
    int device = 0;
    cudaDeviceProp prop{};
    ConeUtil::checkCudaErrorCode(cudaGetDevice(&device));
    ConeUtil::checkCudaErrorCode(cudaGetDeviceProperties(&prop, device));
    const std::string tag = "_sm" + std::to_string(prop.major) + std::to_string(prop.minor) +
                            "_trt" + std::to_string(getInferLibVersion());
    return insertBeforeExtension(path, tag);
}
/**
 * INT8 calibrator over an image folder. Batch 1 (the engine profile is
 * fixed to batch 1; TensorRT reads only one image per getBatch() call for
 * explicit-batch networks), and the same preprocessing as inference via
 * makeInputBlob(). Base: IInt8MinMaxCalibrator or IInt8EntropyCalibrator2.
 */
template <class Base>
class ImageCalibrator : public Base {
public:
    ImageCalibrator(std::vector<std::string> images, int height, int width, std::string cacheFile, std::string inputName)
        : m_images(std::move(images)), m_height(height), m_width(width), m_cacheFile(std::move(cacheFile)),
          m_inputName(std::move(inputName)) {}

    int32_t getBatchSize() const noexcept override { return 1; }

    bool getBatch(void *bindings[], char const *names[], int32_t nbBindings) noexcept override {
        try {
            while (m_next < m_images.size()) {
                const std::string &path = m_images[m_next++];
                cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
                if (img.empty()) {
                    RCLCPP_WARN(logger(), "Calibration: cannot read %s, skipped", path.c_str());
                    continue;
                }
                if (m_next % 100 == 0 || m_next == m_images.size()) {
                    RCLCPP_INFO(logger(), "Calibration: %zu/%zu images", m_next, m_images.size());
                }
                m_gpu.upload(img);
                makeInputBlob(m_gpu, m_height, m_width, m_blob);
                // OpenCV CUDA calls above are synchronous (default stream)
                if (nbBindings < 1 || m_inputName != names[0]) {
                    RCLCPP_ERROR(logger(), "Calibration: unexpected input binding");
                    return false;
                }
                bindings[0] = m_blob.ptr<float>();
                return true;
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(logger(), "Calibration: %s", e.what());
        }
        return false;
    }

    void const *readCalibrationCache(std::size_t &length) noexcept override {
        m_cache.clear();
        std::ifstream in(m_cacheFile, std::ios::binary);
        if (in.good()) {
            m_cache.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            RCLCPP_INFO(logger(), "Calibration: using cache %s (delete it to recalibrate)", m_cacheFile.c_str());
        }
        length = m_cache.size();
        return m_cache.empty() ? nullptr : m_cache.data();
    }

    void writeCalibrationCache(void const *ptr, std::size_t length) noexcept override {
        std::ofstream out(m_cacheFile, std::ios::binary);
        out.write(static_cast<const char *>(ptr), static_cast<std::streamsize>(length));
        RCLCPP_INFO(logger(), "Calibration cache written: %s", m_cacheFile.c_str());
    }

private:
    std::vector<std::string> m_images;
    int m_height;
    int m_width;
    std::string m_cacheFile;
    std::string m_inputName;
    size_t m_next = 0;
    cv::cuda::GpuMat m_gpu;
    cv::cuda::GpuMat m_blob;
    std::vector<char> m_cache;
};

/**
 * Module prefix of the YOLOv8 Detect head, taken from the last layer
 * ("/model.22/Concat_5" -> "/model.22/"). Empty if the names do not follow
 * the Ultralytics export pattern.
 */
std::string detectHeadPrefix(const nvinfer1::INetworkDefinition &network) {
    if (network.getNbLayers() == 0) return "";
    const std::string last = network.getLayer(network.getNbLayers() - 1)->getName();
    if (last.size() < 2 || last[0] != '/') return "";
    const auto slash = last.find('/', 1);
    return slash == std::string::npos ? "" : last.substr(0, slash + 1);
}

/** Detect-head post-processing layer (DFL, box decode, sigmoid, concat), not the per-scale convs. */
bool isDetectHeadPostprocess(const std::string &name, const std::string &prefix) {
    return !prefix.empty() && name.rfind(prefix, 0) == 0 &&
           name.find("/cv2.") == std::string::npos && name.find("/cv3.") == std::string::npos;
}
}  // namespace

void TrtModel::keepDetectHeadFp32(nvinfer1::INetworkDefinition &network, nvinfer1::IBuilderConfig &config) {
    // The YOLOv8 output concatenates box coordinates (0..input size, pixels)
    // and class scores (0..1). In INT8 they share one scale, so the scores
    // are rounded to 0 and most detections disappear. Keep the head
    // post-processing in FP32; the backbone and neck stay INT8.
    const std::string prefix = detectHeadPrefix(network);
    int pinned = 0;
    for (int i = 0; i < network.getNbLayers(); ++i) {
        nvinfer1::ILayer *layer = network.getLayer(i);
        if (!isDetectHeadPostprocess(layer->getName(), prefix)) continue;
        // Shape / constant layers have no precision choice
        if (layer->getType() == nvinfer1::LayerType::kCONSTANT || layer->getType() == nvinfer1::LayerType::kSHAPE) continue;
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        for (int o = 0; o < layer->getNbOutputs(); ++o) {
            if (layer->getOutput(o)->getType() == nvinfer1::DataType::kFLOAT) {
                layer->setOutputType(o, nvinfer1::DataType::kFLOAT);
            }
        }
        ++pinned;
    }
    if (pinned == 0) {
        RCLCPP_WARN(logger(), "int8_keep_head_fp32: no YOLOv8 detect-head layers found, whole network is INT8");
        return;
    }
    config.setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    RCLCPP_INFO(logger(), "INT8: %d detect-head layers (%s*) kept in FP32", pinned, prefix.c_str());
}

void Logger::log(Severity severity, const char *msg) noexcept {
    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

TrtModel::TrtModel(const ModelParams& params) : Detector(params) {
    switch (m_config.precision) {
        case Precision::FP32: m_config.enginePath = params.fp32EnginePath; break;
        case Precision::FP16: m_config.enginePath = params.enginePath; break;
        case Precision::INT8: m_config.enginePath = params.quantizedEnginePath; break;
    }
    if (m_config.enginePath.empty()) {
        throw std::runtime_error("No engine path configured for precision " +
                                 precisionToString(m_config.precision));
    }
    if (m_config.precision == Precision::INT8) {
        // Settings that change the INT8 engine go into its name, so an A/B
        // run never reuses an engine (or calibration cache, named after the
        // engine) built with other settings: "_minmax_n1000" / "_entropy_n300_headint8"
        std::string tag = "_" + m_config.calibrationAlgorithm + "_n" + std::to_string(m_config.calibrationMaxImages);
        if (!m_config.int8KeepHeadFp32) tag += "_headint8";
        m_config.enginePath = insertBeforeExtension(m_config.enginePath, tag);
    }
    if (m_config.tagEngineWithPlatform) {
        m_config.enginePath = tagEnginePath(m_config.enginePath);
    }
    RCLCPP_INFO(logger(), "TensorRT %d, precision %s, engine %s", getInferLibVersion(),
                precisionToString(m_config.precision).c_str(), m_config.enginePath.c_str());

    // The destructor does not run when the constructor throws: free the
    // output buffers and the stream here on failure.
    try {
        if (!loadEngine()) {
            if (!buildEngine()) {
                throw std::runtime_error("Failed to build and load engine.");
            }
        }

        // Create the cuda stream that will be used for inference
        ConeUtil::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
        warmUp();
    } catch (...) {
        try { clearGpuBuffers(); } catch (...) {}
        if (inferenceCudaStream) {
            cudaStreamDestroy(inferenceCudaStream);
            inferenceCudaStream = nullptr;
        }
        throw;
    }
}

TrtModel::~TrtModel() {
    // Destructors must not throw: report CUDA errors instead of checkCudaErrorCode
    try {
        clearGpuBuffers();
    } catch (const std::exception &e) {
        RCLCPP_ERROR(logger(), "Error freeing GPU buffers: %s", e.what());
    }
    if (inferenceCudaStream) {
        cudaStreamDestroy(inferenceCudaStream);
    }
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}

bool TrtModel::loadEngine() {
    std::ifstream file(m_config.enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        RCLCPP_WARN(logger(), "Engine file does not exist or cannot be opened: %s", m_config.enginePath.c_str());
        return false;
    }
    RCLCPP_INFO(logger(), "Loading engine from: %s", m_config.enginePath.c_str());
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        RCLCPP_ERROR(logger(), "Unable to read engine file: %s", m_config.enginePath.c_str());
        return false;
    }

    // Release a previous engine (reload after build). TensorRT requires the
    // context and the buffers to go before the engine that owns them.
    clearGpuBuffers();
    m_context.reset();
    m_engine.reset();

    // Create a runtime to deserialize the engine file.
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
    if (!m_runtime) {
        return false;
    }

    // Fails for engines built by another TensorRT version or GPU; the caller
    // then rebuilds the engine from ONNX.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        RCLCPP_WARN(logger(), "Engine %s cannot be deserialized on this platform, rebuilding.",
                    m_config.enginePath.c_str());
        return false;
    }

    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }

    m_buffers.assign(m_engine->getNbIOTensors(), nullptr);

    m_outputLengths.clear();
    m_outputElementSizes.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_IOTensorNames.clear();

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* tensorName = m_engine->getIOTensorName(i);
        m_IOTensorNames.emplace_back(tensorName);
        const nvinfer1::TensorIOMode tensorType = m_engine->getTensorIOMode(tensorName);
        const nvinfer1::Dims tensorShape = m_engine->getTensorShape(tensorName);
        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            // runInference() binds the input blob to buffer 0
            if (i != 0) {
                RCLCPP_ERROR(logger(), "Only single-input engines with the input at index 0 are supported");
                return false;
            }
            if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                RCLCPP_ERROR(logger(), "Error, the implementation currently only supports float inputs");
                return false;
            }
            m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
            m_inputBatchSize = tensorShape.d[0];
        } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            uint32_t outputLength = 1;
            m_outputDims.push_back(tensorShape);

            for (int j = 1; j < tensorShape.nbDims; ++j) {
                // j = 0 is the batch size (always 1)
                outputLength *= tensorShape.d[j];
            }

            const nvinfer1::DataType outputDataType = m_engine->getTensorDataType(tensorName);
            size_t elementSize = sizeof(float);
            switch (outputDataType) {
                case nvinfer1::DataType::kFLOAT:  elementSize = sizeof(float);    break;
                case nvinfer1::DataType::kHALF:   elementSize = sizeof(__half);   break;
                case nvinfer1::DataType::kINT8:   elementSize = sizeof(int8_t);   break;
                case nvinfer1::DataType::kINT32:  elementSize = sizeof(int32_t);  break;
                default:
                    RCLCPP_ERROR(logger(), "Unsupported output data type for tensor: %s", tensorName);
                    return false;
            }
            m_outputElementSizes.push_back(elementSize);
            m_outputLengths.push_back(outputLength);
            ConeUtil::checkCudaErrorCode(cudaMalloc(&m_buffers[i], outputLength * elementSize));
        } else {
            RCLCPP_ERROR(logger(), "Error, IO Tensor is neither an input or output!");
            return false;
        }
    }

    if (m_inputDims.empty() || m_outputDims.empty()) {
        RCLCPP_ERROR(logger(), "Engine must have one input and at least one output");
        return false;
    }

    // Shared pre/postprocessing buffers (Detector). Throws on a class
    // count mismatch, which the caller reports.
    const nvinfer1::Dims3 &inDims = m_inputDims[0];
    setInputSize(inDims.d[1], inDims.d[2]);
    try {
        setOutputShape(m_outputDims[0].d[1], m_outputDims[0].d[2]);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(logger(), "%s", e.what());
        return false;
    }
    return true;
}

bool TrtModel::buildEngine() {
    RCLCPP_INFO(logger(), "Building %s engine from: %s", precisionToString(m_config.precision).c_str(),
                m_config.onnxModelPath.c_str());
    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    // Explicit batch is the only mode in TensorRT 10 (flag value ignored)
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
        return false;
    }

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    std::ifstream file(m_config.onnxModelPath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        RCLCPP_ERROR(logger(), "ONNX model does not exist or cannot be opened: %s", m_config.onnxModelPath.c_str());
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        RCLCPP_ERROR(logger(), "Unable to read ONNX model: %s", m_config.onnxModelPath.c_str());
        return false;
    }

    if (!parser->parse(buffer.data(), buffer.size())) {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
            RCLCPP_ERROR(logger(), "ONNX parser: %s", parser->getError(i)->desc());
        }
        return false;
    }

    const int32_t numInputs = network->getNbInputs();
    if (numInputs < 1) {
        RCLCPP_ERROR(logger(), "Error, model needs at least 1 input!");
        return false;
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    // Single optimization profile, batch size 1
    nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    for (int32_t i = 0; i < numInputs; ++i) {
        const nvinfer1::ITensor* input = network->getInput(i);
        const char* inputName = input->getName();
        const nvinfer1::Dims inputDims = input->getDimensions();
        const nvinfer1::Dims4 dims(1, inputDims.d[1], inputDims.d[2], inputDims.d[3]);
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, dims);
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, dims);
        optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, dims);
    }
    config->addOptimizationProfile(optProfile);

    switch (m_config.precision) {
        case Precision::FP32:
            break;
        case Precision::FP16:
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            break;
        case Precision::INT8: {
            if (numInputs > 1) {
                RCLCPP_ERROR(logger(), "INT8 quantization is only supported for single input models");
                return false;
            }
            if (m_config.calibrationDataPath.empty()) {
                RCLCPP_ERROR(logger(), "INT8 precision requires calibration_data_path");
                return false;
            }
            // Implicit (calibrator based) INT8 is deprecated since TensorRT
            // 10.1 but still supported in 10.x. platformHasFastInt8() is
            // deprecated too and is not checked: every GPU supported by
            // CUDA 13 has INT8 tensor cores.
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            if (m_config.int8Fp16Fallback) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }

            const nvinfer1::ITensor* input = network->getInput(0);
            const nvinfer1::Dims inputDims = input->getDimensions();
            std::vector<std::string> images = calibrationImages(
                m_config.calibrationDataPath, static_cast<size_t>(std::max(0, m_config.calibrationMaxImages)));
            if (images.empty()) {
                RCLCPP_ERROR(logger(), "No calibration images in %s", m_config.calibrationDataPath.c_str());
                return false;
            }
            // Calibration cache next to the engine; the engine name already
            // holds algorithm and image count. Delete it to recalibrate.
            const std::string cacheFile = m_config.enginePath + ".calibration";
            RCLCPP_INFO(logger(), "INT8 calibration: %s, %zu images from %s", m_config.calibrationAlgorithm.c_str(),
                        images.size(), m_config.calibrationDataPath.c_str());
            if (m_config.calibrationAlgorithm == "minmax") {
                m_calibrator = std::make_unique<ImageCalibrator<nvinfer1::IInt8MinMaxCalibrator>>(
                    std::move(images), inputDims.d[2], inputDims.d[3], cacheFile, input->getName());
            } else if (m_config.calibrationAlgorithm == "entropy") {
                m_calibrator = std::make_unique<ImageCalibrator<nvinfer1::IInt8EntropyCalibrator2>>(
                    std::move(images), inputDims.d[2], inputDims.d[3], cacheFile, input->getName());
            } else {
                RCLCPP_ERROR(logger(), "calibration_algorithm must be minmax or entropy, got %s",
                             m_config.calibrationAlgorithm.c_str());
                return false;
            }
            config->setInt8Calibrator(m_calibrator.get());

            if (m_config.int8KeepHeadFp32) {
                keepDetectHeadFp32(*network, *config);
            }
            break;
        }
    }

    // CUDA stream used for profiling by the builder
    cudaStream_t profileStream = nullptr;
    ConeUtil::checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    cudaStreamDestroy(profileStream);
    m_calibrator.reset();
    if (!plan) {
        RCLCPP_ERROR(logger(), "TensorRT engine build failed");
        return false;
    }

    std::ofstream outfile(m_config.enginePath, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    outfile.close();
    if (!outfile) {
        RCLCPP_ERROR(logger(), "Unable to write engine file: %s", m_config.enginePath.c_str());
        return false;
    }
    RCLCPP_INFO(logger(), "Engine written to: %s", m_config.enginePath.c_str());

    return loadEngine();
}

void TrtModel::clearGpuBuffers() {
    if (m_buffers.empty()) {
        return;
    }
    // Only output buffers are owned. Buffer 0 (input) points into `blob`.
    for (size_t i = 0; i < m_buffers.size(); ++i) {
        if (m_engine && i < m_IOTensorNames.size() &&
            m_engine->getTensorIOMode(m_IOTensorNames[i].c_str()) == nvinfer1::TensorIOMode::kOUTPUT &&
            m_buffers[i] != nullptr) {
            ConeUtil::checkCudaErrorCode(cudaFree(m_buffers[i]));
        }
    }
    m_buffers.clear();
}

std::string TrtModel::runtimeVersion() const {
    return "tensorrt=" + std::to_string(getInferLibVersion());
}

bool TrtModel::infer(const cv::cuda::GpuMat &blob, std::vector<float> &featureVector) {
    const nvinfer1::Dims3 &dims = m_inputDims[0];
    const size_t inputCount = static_cast<size_t>(dims.d[0]) * dims.d[1] * dims.d[2];
    if (blob.empty() || blob.type() != CV_32FC3 || !blob.isContinuous() ||
        static_cast<size_t>(blob.rows) * blob.cols * blob.channels() != inputCount) {
        RCLCPP_ERROR(logger(), "Input blob does not have the engine input size!");
        return false;
    }

    nvinfer1::Dims4 inputDims = {1, dims.d[0], dims.d[1], dims.d[2]};
    m_context->setInputShape(m_IOTensorNames[0].c_str(), inputDims);

    // CHW float blob from Detector::preprocess(), already on the device
    m_buffers[0] = const_cast<void *>(static_cast<const void *>(blob.ptr<float>()));

    if (!m_context->allInputDimensionsSpecified()) {
        RCLCPP_ERROR(logger(), "Error, not all required dimensions specified.");
        return false;
    }

    for (size_t i = 0; i < m_buffers.size(); ++i) {
        if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
            return false;
        }
    }

    if (!m_context->enqueueV3(inferenceCudaStream)) {
        return false;
    }

    // Only the first output (YOLO detect head) is decoded
    const uint32_t outputLength = m_outputLengths[0];
    const size_t elementSize = m_outputElementSizes[0];
    featureVector.resize(outputLength);
    if (elementSize == sizeof(float)) {
        ConeUtil::checkCudaErrorCode(cudaMemcpyAsync(featureVector.data(), m_buffers[1],
                                                     outputLength * sizeof(float), cudaMemcpyDeviceToHost,
                                                     inferenceCudaStream));
        ConeUtil::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    } else {
        // Non-float output: copy raw bytes, then convert to float
        m_rawOutputBuffer.resize(outputLength * elementSize);
        ConeUtil::checkCudaErrorCode(cudaMemcpyAsync(m_rawOutputBuffer.data(), m_buffers[1],
                                                     outputLength * elementSize, cudaMemcpyDeviceToHost,
                                                     inferenceCudaStream));
        ConeUtil::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        const nvinfer1::DataType type = m_engine->getTensorDataType(m_IOTensorNames[1].c_str());
        if (type == nvinfer1::DataType::kHALF) {
            const __half* src = reinterpret_cast<const __half*>(m_rawOutputBuffer.data());
            for (uint32_t k = 0; k < outputLength; ++k) {
                featureVector[k] = __half2float(src[k]);
            }
        } else if (type == nvinfer1::DataType::kINT8) {
            // Raw INT8 values without the tensor scale; only meaningful if
            // the engine was built with an INT8 output on purpose.
            const int8_t* src = reinterpret_cast<const int8_t*>(m_rawOutputBuffer.data());
            for (uint32_t k = 0; k < outputLength; ++k) {
                featureVector[k] = static_cast<float>(src[k]);
            }
        } else if (type == nvinfer1::DataType::kINT32) {
            const int32_t* src = reinterpret_cast<const int32_t*>(m_rawOutputBuffer.data());
            for (uint32_t k = 0; k < outputLength; ++k) {
                featureVector[k] = static_cast<float>(src[k]);
            }
        }
    }
    return true;
}
