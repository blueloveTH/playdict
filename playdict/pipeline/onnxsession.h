#ifndef ONNXSESSION_H
#define ONNXSESSION_H

#include "onnxruntime_cxx_api.h"

class ONNXSession{
    Ort::Session *session;
public:
    ONNXSession(const char* name, const wchar_t* model_path){
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, name);

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session = new Ort::Session(env, model_path, session_options);
    }

    std::vector<int64_t> inputShape(){
        return session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    }

    template<typename T>
    Ort::Value createTensor(T *data, const std::vector<int64_t>& shape){
        size_t size = 1;
        for(auto s : shape) size *= s;
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return Ort::Value::CreateTensor<T>(memoryInfo, data, size, shape.data(), shape.size());
    }

    auto run(Ort::Value *x, std::vector<const char*> outputNames){
        std::vector<const char*> input_node_names = {"x"};
        return session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), x, 1, outputNames.data(), outputNames.size());
    }
};

#endif // ONNXSESSION_H
