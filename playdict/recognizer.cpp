#include "recognizer.h"

Recognizer::Recognizer(QObject *parent) : QObject(parent)
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const wchar_t* model_path = L"vgg_lstm_quantized.onnx";
    session = new Ort::Session(env, model_path, session_options);
}

QString Recognizer::model_predict(const QPixmap& map){
    auto img = map.toImage();
    img.convertTo(QImage::Format_Grayscale8);
    img = img.scaled(128, 32);

    size_t input_size = 1 * 1 * 32 * 128;
    std::vector<float> x_test(input_size);

    for(uint i=0; i<input_size; i++){
        float pix = (float)img.constBits()[i];
        pix = (pix/255.0 - 0.449) / 0.226;
        x_test[i] = pix;
    }

    auto input_dims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, x_test.data(), input_size, input_dims.data(), 4);

    std::vector<const char*> input_node_names = {"x"};
    std::vector<const char*> output_node_names = {"sequence"};

    auto output_list = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

    int* seq = output_list.front().GetTensorMutableData<int>();
    int element_cnt = output_list.front().GetTensorTypeAndShapeInfo().GetElementCount();

    QString mapping = "???0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -";

    QString word = "";

    for(int i=0; i<element_cnt; i++){
        if(seq[i] < 2) continue;
        if(seq[i] == 2) break;
        word += mapping[seq[i]];
    }
    return word;
}

void Recognizer::exec(QPixmap map){
    if(!_isReady)
        return;
    _isReady = false;

    QtConcurrent::run([=]{
        QString word = model_predict(map);
        _isReady = true;
        emit finished(word);
    });
}
