#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <QPixmap>
#include <QImage>
#include "onnxsession.h"

class Recognizer
{
    ONNXSession *session;

public:
    Recognizer(){
        session = new ONNXSession("recognizer", L"models/vgg_lstm_quantized.onnx");
    }

    QString predict(QImage img){
        img.convertTo(QImage::Format_Grayscale8);
        img = img.scaled(128, 32);

        Ort::Value inputTensor = session->createTensor<uchar>(img.bits(), std::vector<int64_t>{1,1,32,128});
        auto oList = session->run(&inputTensor, std::vector<const char*>{"sequence"});

        int* seq = oList.front().GetTensorMutableData<int>();
        int elementCnt = (int)oList.front().GetTensorTypeAndShapeInfo().GetElementCount();

        QString mapping = "???0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -";

        QString word = "";

        for(int i=0; i<elementCnt; i++){
            if(seq[i] < 2) continue;
            if(seq[i] == 2) break;
            word += mapping[seq[i]];
        }

        return word;
    }
};

#endif // RECOGNIZER_H
