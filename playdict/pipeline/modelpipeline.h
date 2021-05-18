#ifndef MODELPIPELINE_H
#define MODELPIPELINE_H

#include <QtConcurrent/QtConcurrentRun>
#include "bingdict.h"
#include "recognizer.h"

class ModelPipeline : public QObject{

    Q_OBJECT

    Recognizer recognizer;
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

            QString word = recognizer.predict(img);

            timeList.append(clock());

            WordInfo wi = bingDict.query(word);

            timeList.append(clock());

            emit finished(wi);

            clock_t cost_0 = timeList[1] - timeList[0];
            clock_t cost_1 = timeList[2] - timeList[1];
            qDebug() << QString("Re: %1ms, Se: %2ms").arg(cost_0).arg(cost_1);

            _isReady = true;
        });
    }
};

#endif // MODELPIPELINE_H
