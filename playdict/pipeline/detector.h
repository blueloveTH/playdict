#ifndef DETECTOR_H
#define DETECTOR_H

#include "onnxsession.h"
#include "boxalgorithm.h"
#include <QImage>

class Detector
{
    ONNXSession *session;

public:
    Detector(){
        session = new ONNXSession(":/models/res/pooling.onnx");
    }

    QImage crop(QImage img){
        img = img.scaled(144*2, 32*2, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        img.convertTo(QImage::Format_Grayscale8);

        Ort::Value inputTensor = session->createTensor<uchar>(img.bits(), std::vector<int64_t>{1,1,32*2,144*2});
        auto oList = session->run(&inputTensor);

        const int* output = oList[0].GetTensorMutableData<int>();
        Rect rect = getMainBox(MatArray(output, 32*2, 144*2));

        if(rect.isEmpty()) return QImage();

        /*QImage img_2(img);
        for(int i=0; i<32*2; i++){
            for(int j=0; j<144*2; j++){
                uint v = output[i*144*2+j] * 255;
                img_2.setPixel(j, i, qRgb(v,v,v));
            }
        }
        img_2.save("123.png");*/

        img = img.copy(rect.yMin, rect.xMin, rect.ySpan(), rect.xSpan());
        img = img.scaled(144, 32, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        img.convertTo(QImage::Format_Grayscale8);
        return img;
    }

};

#endif // DETECTOR_H
