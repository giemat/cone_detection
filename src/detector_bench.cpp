// Offline detector benchmark: runs one backend + precision over a folder of
// images and writes the same <tag>_frames.csv / <tag>_detections.csv files as
// the cone_detection node, so scripts/benchmark/summarize.py reads both.
//
// Measures only the detector (upload, preprocess, inference, postprocess);
// no ROS, no lidar fusion. Use it to compare backends and precisions on
// labelled or calibration images; use run_benchmark.sh for the full pipeline.
//
//   ros2 run cone_detection detector_bench --backend onnx --precision fp16
//       --model ~/ws/src/cone_detection/models/Yolov8n1280.onnx
//       --images ~/datasets/val --out ~/results/x86_64_offline --run 1
#include "cone_detection/detector.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace {
using Clock = std::chrono::steady_clock;

int64_t elapsedUs(const Clock::time_point &start, const Clock::time_point &end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

const char *kUsage = R"(Usage: detector_bench --model <fp32.onnx> --images <dir> --out <dir> [options]

  --backend tensorrt|onnx   (default tensorrt)
  --precision fp32|fp16|int8 (default fp32)
  --model PATH              FP32 ONNX model. Derived from it (same folder, same
                            names as the launch file `model:=` argument):
                              TensorRT  <name>_FP32.engine, <name>.engine (fp16),
                                        <name>_INT8.engine (platform tag appended)
                              ONNX RT   <name>_fp16.onnx, <name>_int8_qdq.onnx
  --engine-dir DIR          where TensorRT engines are read / written (default: model folder)
  --calib DIR               INT8 calibration images (TensorRT int8 build)
  --images DIR              .jpg/.png images, processed in sorted order
  --max-images N            0 = all (default)
  --warmup N                frames run before logging (default 20)
  --out DIR                 output folder (created)
  --run N                   run index for the file name (default 1)
  --label TEXT              free text for the CSV header
  --classes a,b,c           class names (default: the 10 cone classes of params.yaml)
  --conf X / --iou X        thresholds (default 0.25 / 0.5, as params.yaml)
  --cuda-plugin PATH        ONNX Runtime CUDA plugin EP (aarch64)
)";

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::string stem(const std::string &path) {
    return fs::path(path).stem().string();
}
}  // namespace

int main(int argc, char **argv) {
    std::map<std::string, std::string> a = {
        {"backend", "tensorrt"}, {"precision", "fp32"}, {"max-images", "0"}, {"warmup", "20"},
        {"run", "1"}, {"conf", "0.25"}, {"iou", "0.5"},
        {"classes", "seg_orange_cone,unknown_cone,yellow_cone,seg_large_orange_cone,seg_blue_cone,"
                    "seg_unknown_cone,seg_yellow_cone,blue_cone,orange_cone,large_orange_cone"},
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "-h" || k == "--help" || k.rfind("--", 0) != 0 || i + 1 >= argc) {
            std::cerr << kUsage;
            return k == "-h" || k == "--help" ? 0 : 1;
        }
        a[k.substr(2)] = argv[++i];
    }
    for (const char *req : {"model", "images", "out"}) {
        if (!a.count(req)) {
            std::cerr << "missing --" << req << "\n" << kUsage;
            return 1;
        }
    }

    const fs::path model(a["model"]);
    const std::string name = stem(a["model"]);
    const fs::path dir = model.parent_path();
    const fs::path engineDir = a.count("engine-dir") ? fs::path(a["engine-dir"]) : dir;

    ModelParams p;
    p.backend = backendFromString(a["backend"]);
    p.precision = precisionFromString(a["precision"]);
    p.onnxModelPath = model.string();
    p.fp32EnginePath = (engineDir / (name + "_FP32.engine")).string();
    p.enginePath = (engineDir / (name + ".engine")).string();
    p.quantizedEnginePath = (engineDir / (name + "_INT8.engine")).string();
    p.calibrationDataPath = a.count("calib") ? a["calib"] : "";
    p.onnxFp16ModelPath = (dir / (name + "_fp16.onnx")).string();
    p.onnxInt8ModelPath = (dir / (name + "_int8_qdq.onnx")).string();
    p.onnxCudaPluginPath = a.count("cuda-plugin") ? a["cuda-plugin"] : "";
    p.classes = split(a["classes"], ',');
    p.rect_confidence_threshold = std::stof(a["conf"]);
    p.iou_threshold = std::stof(a["iou"]);

    std::vector<std::string> images;
    for (const auto &e : fs::directory_iterator(a["images"])) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            images.push_back(e.path().string());
        }
    }
    std::sort(images.begin(), images.end());
    const size_t maxImages = std::stoul(a["max-images"]);
    if (maxImages > 0 && images.size() > maxImages) {
        images.resize(maxImages);
    }
    if (images.empty()) {
        std::cerr << "no images in " << a["images"] << "\n";
        return 1;
    }

    std::unique_ptr<Detector> det;
    try {
        det = makeDetector(p);
    } catch (const std::exception &e) {
        std::cerr << "detector init failed: " << e.what() << "\n";
        return 2;
    }

    fs::create_directories(a["out"]);
    const std::string tag = name + "_" + det->backend() + "_" + precisionToString(det->precision()) + "_run" + a["run"];
    const fs::path outDir(a["out"]);
    std::ofstream frames(outDir / (tag + "_frames.csv"));
    std::ofstream dets(outDir / (tag + "_detections.csv"));
    const int warmup = std::stoi(a["warmup"]);
    frames << "# label=" << (a.count("label") ? a["label"] : "") << "\n"
           << "# source=detector_bench\n"
           << "# backend=" << det->backend() << "\n"
           << "# precision=" << precisionToString(det->precision()) << "\n"
           << "# engine=" << det->modelPath() << "\n"
           << "# runtime=" << det->runtimeVersion() << "\n"
#ifdef NDEBUG
           << "# build=release\n"
#else
           << "# build=debug\n"
#endif
           << "# images=" << a["images"] << "\n"
           << "# warmup_frames=" << warmup << "\n";
    frames << "frame,stamp_ns,upload_us,detection_time_us,model_total_us,preprocess_us,inference_us,"
              "postprocess_us,filter_time_us,fusion_time_us,callback_total_us,raw_cones,filtered_cones,"
              "fused_candidates,published_cones\n";
    dets << "frame,stamp_ns,class_id,class_name,confidence,x,y,w,h\n";

    // Warm-up on the first images (not logged), then one pass over all images.
    // stamp_ns = image index: the same image gets the same key in every run,
    // which is what summarize.py matches on.
    cv::cuda::GpuMat gpu;
    for (int i = 0; i < warmup; ++i) {
        cv::Mat img = cv::imread(images[i % images.size()], cv::IMREAD_COLOR);
        gpu.upload(img);
        det->detect(gpu);
    }
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat img = cv::imread(images[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "skip unreadable " << images[i] << "\n";
            continue;
        }
        const auto t0 = Clock::now();
        gpu.upload(img);
        const auto t1 = Clock::now();
        const std::vector<ModelResult> res = det->detect(gpu);
        const auto t2 = Clock::now();
        const ModelTimings &t = det->lastTimings();
        const int64_t upload = elapsedUs(t0, t1);
        frames << i << ',' << i << ',' << upload << ',' << elapsedUs(t1, t2) << ',' << t.total_us << ','
               << t.preprocess_us << ',' << t.inference_us << ',' << t.postprocess_us << ",0,0,"
               << elapsedUs(t0, t2) << ',' << res.size() << ',' << res.size() << ",0," << res.size() << '\n';
        for (const auto &r : res) {
            dets << i << ',' << i << ',' << r.class_id << ',' << det->get_class_by_id(r.class_id) << ','
                 << r.confidence << ',' << r.box.x << ',' << r.box.y << ',' << r.box.width << ','
                 << r.box.height << '\n';
        }
        if ((i + 1) % 500 == 0) {
            std::cout << tag << ": " << (i + 1) << "/" << images.size() << std::endl;
        }
    }
    std::cout << tag << ": " << images.size() << " images -> " << a["out"] << std::endl;
    return 0;
}
