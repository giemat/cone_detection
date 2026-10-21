#include "cone_detection/model_onnx.hpp"

#include <mutex>
#include <unordered_map>

namespace {
rclcpp::Logger logger() {
    return rclcpp::get_logger("cone_detection");
}

std::string shapeToString(const std::vector<int64_t> &shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return s + "]";
}

size_t elementCount(const std::vector<int64_t> &shape) {
    size_t n = 1;
    for (int64_t d : shape) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

constexpr const char *kPluginRegistrationName = "cone_detection_cuda_plugin";
constexpr const char *kCudaEpName = "CUDAExecutionProvider";
// Where Dockerfile.jetpack unpacks the CUDA plugin EP on aarch64
constexpr const char *kDefaultPluginPath = "/opt/onnxruntime/plugin-ep-cuda/libonnxruntime_providers_cuda.so";
}  // namespace

namespace {
std::mutex g_envMutex;
std::weak_ptr<Ort::Env> g_env;
// Plugin EP libraries are registered per Env
bool g_pluginRegistered = false;
}  // namespace

std::shared_ptr<Ort::Env> OnnxModel::sharedEnv() {
    std::lock_guard<std::mutex> lock(g_envMutex);
    std::shared_ptr<Ort::Env> env = g_env.lock();
    if (!env) {
        env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "cone_detection");
        g_env = env;
        g_pluginRegistered = false;
    }
    return env;
}

OnnxModel::OnnxModel(const ModelParams &params) : Detector(params) {
    switch (m_config.precision) {
        case Precision::FP32: m_modelPath = params.onnxModelPath; break;
        case Precision::FP16: m_modelPath = params.onnxFp16ModelPath; break;
        case Precision::INT8: m_modelPath = params.onnxInt8ModelPath; break;
    }
    if (m_modelPath.empty()) {
        throw std::runtime_error("No ONNX model configured for precision " + precisionToString(m_config.precision) +
                                 " (model_path / onnx.fp16_model_path / onnx.int8_model_path)");
    }
    if (!ConeUtil::doesFileExist(m_modelPath)) {
        throw std::runtime_error("ONNX model not found: " + m_modelPath +
                                 ". fp16/int8 files are made by scripts/benchmark/export_onnx_variants.py");
    }
#if defined(__aarch64__)
    // The aarch64 ONNX Runtime package is CPU only: use the plugin by default
    if (m_config.onnxCudaPluginPath.empty() && ConeUtil::doesFileExist(kDefaultPluginPath)) {
        m_config.onnxCudaPluginPath = kDefaultPluginPath;
    }
#endif
    m_usePlugin = !m_config.onnxCudaPluginPath.empty();

    // The Env registers ONNX Runtime's default logger; it must exist before
    // any other ORT object (CUDA provider options fail without it).
    m_env = sharedEnv();
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(params.onnxIntraOpThreads);
    options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    if (params.onnxDisableCpuFallback) {
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
    appendCudaExecutionProvider(options);

    RCLCPP_INFO(logger(), "ONNX Runtime %s (%s), precision %s, model %s", Ort::GetVersionString().c_str(),
                m_usePlugin ? "CUDA plugin EP" : "CUDA EP", precisionToString(m_config.precision).c_str(),
                m_modelPath.c_str());
    m_session = std::make_unique<Ort::Session>(*m_env, m_modelPath.c_str(), options);

    if (m_session->GetInputCount() != 1 || m_session->GetOutputCount() < 1) {
        throw std::runtime_error("ONNX model must have one input and at least one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    m_inputName = m_session->GetInputNameAllocated(0, allocator).get();
    m_outputName = m_session->GetOutputNameAllocated(0, allocator).get();

    // Input [1, 3, H, W] float. FP16 models must keep float32 I/O
    // (convert_float_to_float16(keep_io_types=True)), so the blob is the same.
    // Keep the TypeInfo alive: the shape info is a view into it
    const Ort::TypeInfo inputType = m_session->GetInputTypeInfo(0);
    const auto inputInfo = inputType.GetTensorTypeAndShapeInfo();
    if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX model input must be float32 (export FP16 models with keep_io_types=True)");
    }
    m_inputShape = inputInfo.GetShape();
    if (m_inputShape.size() != 4 || (m_inputShape[1] != 3 && m_inputShape[1] > 0)) {
        throw std::runtime_error("ONNX model input must be [N, 3, H, W], got " + shapeToString(m_inputShape));
    }
    // Dynamic dimensions: batch 1, size from params.yaml (width, height)
    m_inputShape[0] = 1;
    m_inputShape[1] = 3;
    if (m_inputShape[2] <= 0) m_inputShape[2] = params.img_size.at(1);
    if (m_inputShape[3] <= 0) m_inputShape[3] = params.img_size.at(0);
    m_inputCount = elementCount(m_inputShape);

    const Ort::TypeInfo outputType = m_session->GetOutputTypeInfo(0);
    const auto outputInfo = outputType.GetTensorTypeAndShapeInfo();
    if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX model output must be float32 (export FP16 models with keep_io_types=True)");
    }
    m_outputShape = outputInfo.GetShape();
    if (m_outputShape.size() != 3 || m_outputShape[1] <= 0 || m_outputShape[2] <= 0) {
        throw std::runtime_error("ONNX model output must be a static [1, 4 + classes, N], got " +
                                 shapeToString(m_outputShape));
    }
    m_outputShape[0] = 1;
    m_outputCount = elementCount(m_outputShape);

    setInputSize(static_cast<int>(m_inputShape[2]), static_cast<int>(m_inputShape[3]));
    setOutputShape(static_cast<int32_t>(m_outputShape[1]), static_cast<int32_t>(m_outputShape[2]));

    m_hostMemInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    m_binding = std::make_unique<Ort::IoBinding>(*m_session);
    // Run() synchronises the EP before returning (default run option), so
    // the host output is complete when infer() returns, as for TensorRT.
    m_runOptions = Ort::RunOptions{};

    warmUp();
}

void OnnxModel::appendCudaExecutionProvider(Ort::SessionOptions &options) {
    const std::string &algo = m_config.onnxCudnnConvAlgoSearch;
    if (!m_usePlugin) {
        // Classic CUDA EP, part of the onnxruntime-linux-x64-gpu build
        Ort::CUDAProviderOptions cuda;
        cuda.Update({
            {"device_id", "0"},
            // EXHAUSTIVE: benchmark cuDNN conv algorithms once, the
            // counterpart of the TensorRT tactic search
            {"cudnn_conv_algo_search", algo},
            {"do_copy_in_default_stream", "1"},
        });
        options.AppendExecutionProvider_CUDA_V2(*cuda);
        m_ownedDeviceMemInfo = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        m_deviceMemInfo = m_ownedDeviceMemInfo;
        return;
    }

    // CUDA plugin EP (aarch64). Registered once on the shared Env.
    {
        std::lock_guard<std::mutex> lock(g_envMutex);
        if (!g_pluginRegistered) {
            RCLCPP_INFO(logger(), "Registering CUDA plugin EP: %s", m_config.onnxCudaPluginPath.c_str());
            m_env->RegisterExecutionProviderLibrary(kPluginRegistrationName, m_config.onnxCudaPluginPath);
            g_pluginRegistered = true;
        }
    }

    std::vector<Ort::ConstEpDevice> selected;
    for (const auto &device : m_env->GetEpDevices()) {
        if (std::string(device.EpName()) == kCudaEpName && device.Device().Type() == OrtHardwareDeviceType_GPU) {
            selected.push_back(device);
            break;  // first GPU
        }
    }
    if (selected.empty()) {
        throw std::runtime_error(std::string("No GPU device for ") + kCudaEpName + " after loading " +
                                 m_config.onnxCudaPluginPath);
    }
    try {
        options.AppendExecutionProvider_V2(*m_env, selected,
            std::unordered_map<std::string, std::string>{{"cudnn_conv_algo_search", algo}});
    } catch (const Ort::Exception &e) {
        // Option names of the plugin EP are not fixed yet: fall back to its defaults
        RCLCPP_WARN(logger(), "CUDA plugin EP rejected cudnn_conv_algo_search=%s (%s), using defaults",
                    algo.c_str(), e.what());
        options.AppendExecutionProvider_V2(*m_env, selected, std::unordered_map<std::string, std::string>{});
    }
    // Memory info of the EP's device memory; owned by the EP device (lives as long as m_env)
    m_deviceMemInfo = selected.front().GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    if (m_deviceMemInfo == nullptr) {
        throw std::runtime_error("CUDA plugin EP reports no device memory info");
    }
}

std::string OnnxModel::runtimeVersion() const {
    return "onnxruntime=" + Ort::GetVersionString() + (m_usePlugin ? "+cuda_plugin" : "+cuda");
}

bool OnnxModel::infer(const cv::cuda::GpuMat &blob, std::vector<float> &output) {
    if (blob.empty() || blob.type() != CV_32FC3 || !blob.isContinuous() ||
        static_cast<size_t>(blob.rows) * blob.cols * blob.channels() != m_inputCount) {
        RCLCPP_ERROR(logger(), "Input blob does not have the model input size!");
        return false;
    }
    output.resize(m_outputCount);
    try {
        // Rebind every frame: the blob pointer may change, the output vector too.
        // Creating the Ort::Value wrappers allocates nothing on the device.
        Ort::Value input = Ort::Value::CreateTensor<float>(
            m_deviceMemInfo, const_cast<float *>(blob.ptr<float>()), m_inputCount,
            m_inputShape.data(), m_inputShape.size());
        Ort::Value out = Ort::Value::CreateTensor<float>(
            m_hostMemInfo, output.data(), m_outputCount, m_outputShape.data(), m_outputShape.size());
        m_binding->ClearBoundInputs();
        m_binding->ClearBoundOutputs();
        m_binding->BindInput(m_inputName.c_str(), input);
        // Host output: ORT copies device->host inside Run(), like the
        // cudaMemcpy at the end of TrtModel::infer()
        m_binding->BindOutput(m_outputName.c_str(), out);
        m_session->Run(m_runOptions, *m_binding);
    } catch (const Ort::Exception &e) {
        RCLCPP_ERROR(logger(), "ONNX Runtime inference failed: %s", e.what());
        return false;
    }
    return true;
}
