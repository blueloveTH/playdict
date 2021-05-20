#ifndef ONNXSESSION_H
#define ONNXSESSION_H

#include "onnxruntime_cxx_api.h"
#include <QFile>

class ONNXSession{
    Ort::Session *session;

public:
    ONNXSession(QString qrcPath){
        static Ort::Env *env;
        if(env == nullptr)
            env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "global");

        QFile file(qrcPath);
        file.open(QIODevice::ReadOnly);
        QByteArray model_data = file.readAll();
        file.close();

        Ort::SessionOptions session_options;
        //session_options.SetIntraOpNumThreads(1);
        //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        session = new Ort::Session(*env, model_data.data(), model_data.size(), session_options);
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
        std::vector<const char*> inputNames = {"x"};
        return session->Run(Ort::RunOptions{nullptr}, inputNames.data(), x, inputNames.size(), outputNames.data(), outputNames.size());
    }

    auto run(Ort::Value *x){
        std::vector<const char*> inputNames = {"x"};
        std::vector<const char*> outputNames = {"y"};
        return session->Run(Ort::RunOptions{nullptr}, inputNames.data(), x, inputNames.size(), outputNames.data(), outputNames.size());
    }
};

#endif // ONNXSESSION_H
