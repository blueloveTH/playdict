#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <QPixmap>
#include <QImage>
#include "onnxsession.h"

class Recognizer
{
    ONNXSession *session;
    QString model_path;

public:
    Recognizer(){
        model_path = ":/models/res/vgg_transformer_ctc_quantized.onnx";
        session = new ONNXSession("recognizer", model_path);
    }

    QString predict(QImage img){
        img.convertTo(QImage::Format_Grayscale8);
        img = img.scaled(128, 32);

        Ort::Value inputTensor = session->createTensor<uchar>(img.bits(), std::vector<int64_t>{1,1,32,128});

        std::vector<const char*> outputNames = {"y", "confidence"};
        auto oList = session->run(&inputTensor, outputNames);

        long long* _1 = oList[0].GetTensorMutableData<long long>();
        int elementCnt = (int)oList[0].GetTensorTypeAndShapeInfo().GetElementCount();

        float* conf = oList[1].GetTensorMutableData<float>();

        QString mapping = "?!#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -";

        QList<int> results;
        QString rawWord = "";
        for(int i=0; i<elementCnt; i++){
            rawWord += mapping[(int)_1[i]];
            results.append((int)_1[i]);
        }
        qDebug() << rawWord;

        QString word = "";

        //ctc post processing
        if(model_path.contains("ctc")){
            int curr_i = -1;
            QList<int> ctc_results;
            for(int i : qAsConst(results)){
                if(i != curr_i){
                    ctc_results.append(i);
                    curr_i = i;
                }
            }
            results = ctc_results;
        }

        for(int i=0; i<results.size(); i++){
            if(results[i] < 2) continue;
            //if(results[i] == 2) break;
            word += mapping[results[i]];
            //qDebug() << mapping[results[i]] << conf[i];
        }

        return word;
    }
};

#endif // RECOGNIZER_H
