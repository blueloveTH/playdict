#ifndef MODELPIPELINE_H
#define MODELPIPELINE_H

#include <QtConcurrent/QtConcurrentRun>
#include "bingdict.h"
#include "recognizer.h"
#include "detector.h"

class ModelPipeline : public QObject{

    Q_OBJECT

    Recognizer recognizer;
    Detector detector;

    BingDict bingDict;
    QImage currImg_;

    bool _isReady = true;

signals:
    void finished(const WordInfo&);

public:
    bool isReady(){
        return _isReady;
    }

    QImage currImg(){
        return currImg_;
    }

    void run(QImage img){
        currImg_ = img;

        QtConcurrent::run([=]{
            _isReady = false;

            QList<clock_t> timeList;

            timeList.append(clock());

            uchar* bits = detector.crop(img);

            if(bits == nullptr){
                emit finished(WordInfo("(No word)"));
                _isReady = true;
                return;
            }

            timeList.append(clock());

            QString word = recognizer.predict(bits);

            timeList.append(clock());

            WordInfo wi = bingDict.query(word);

            timeList.append(clock());

            emit finished(wi);

            clock_t cost_de = timeList[1] - timeList[0];
            clock_t cost_re = timeList[2] - timeList[1];
            clock_t cost_se = timeList[3] - timeList[2];
            qDebug() << QString("De: %1ms, Re: %2ms, Se: %3ms").arg(cost_de).arg(cost_re).arg(cost_se);

            _isReady = true;
        });
    }
};

#endif // MODELPIPELINE_H
