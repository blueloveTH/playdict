#include "recognizer.h"

Recognizer::Recognizer(QObject *parent) : QObject(parent)
{
    session = new ONNXSession("recognizer", L"models/vgg_lstm_quantized.onnx");
}

QString Recognizer::model_predict(const QPixmap& map){
    auto img = map.toImage();
    img.convertTo(QImage::Format_Grayscale8);
    img = img.scaled(128, 32);

    Ort::Value inputTensor = session->createTensor<uchar>(img.bits(), std::vector<int64_t>{1,1,32,128});
    auto oList = session->run(&inputTensor, std::vector<const char*>{"sequence"});

    int* seq = oList.front().GetTensorMutableData<int>();
    uint elementCnt = oList.front().GetTensorTypeAndShapeInfo().GetElementCount();

    QString mapping = "???0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -";

    QString word = "";

    for(uint i=0; i<elementCnt; i++){
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
