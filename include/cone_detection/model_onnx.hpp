#pragma once

#include "cone_detection/detector.hpp"

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <string>
#include <vector>

/**
 * ONNX Runtime backend on the CUDA execution provider.
 *
 * Runs the ONNX file for the selected precision (fp32: model_path, fp16 /
 * int8: onnx.fp16_model_path / onnx.int8_model_path). The input is the CHW
 * float blob that Detector::preprocess() leaves on the GPU; it is bound with
 * Ort::IoBinding, so there is no host round trip and pre/postprocessing are
 * the same code as for the TensorRT backend. Only inference differs.
 *
 * CUDA EP selection:
 *   onnx.cuda_plugin_path empty  -> classic CUDA EP of the GPU build (x86)
 *   onnx.cuda_plugin_path set    -> CUDA plugin EP library, registered on the
 *                                   process Ort::Env (aarch64 / JetPack 7)
 */
class OnnxModel : public Detector {
public:
    explicit OnnxModel(const ModelParams &params);
    ~OnnxModel() override = default;

    std::string backend() const override { return "onnx"; }
    std::string modelPath() const override { return m_modelPath; }
    std::string runtimeVersion() const override;

protected:
    bool infer(const cv::cuda::GpuMat &blob, std::vector<float> &output) override;

private:
    /**
     * One Ort::Env per process (ONNX Runtime requirement), shared by all
     * models and freed with the last one. A function-local static would be
     * destroyed at exit, after the CUDA runtime, and crash the process.
     */
    static std::shared_ptr<Ort::Env> sharedEnv();
    void appendCudaExecutionProvider(Ort::SessionOptions &options);

    // Declared first: destroyed after the session and binding
    std::shared_ptr<Ort::Env> m_env;
    std::string m_modelPath;
    bool m_usePlugin = false;
    std::unique_ptr<Ort::Session> m_session;
    std::unique_ptr<Ort::IoBinding> m_binding;
    Ort::RunOptions m_runOptions;
    /** Device memory description for the input tensor (owned for the classic EP). */
    Ort::MemoryInfo m_ownedDeviceMemInfo{nullptr};
    const OrtMemoryInfo *m_deviceMemInfo = nullptr;
    Ort::MemoryInfo m_hostMemInfo{nullptr};
    std::string m_inputName;
    std::string m_outputName;
    std::vector<int64_t> m_inputShape;
    std::vector<int64_t> m_outputShape;
    size_t m_inputCount = 0;
    size_t m_outputCount = 0;
};
