#ifndef MODELPIPELINE_H
#define MODELPIPELINE_H

#include <QtConcurrent/QtConcurrentRun>
#include <QDir>
#include <QDateTime>
#include "bingdict.h"
#include "recognizer.h"
#include "detector.h"
#include "corrector.h"


class ModelPipeline : public QObject{

    Q_OBJECT

    Recognizer recognizer;
    Detector detector;
    Corrector corrector;

    BingDict bingDict;

    bool _isReady = true;
    bool _labelingMode = false;

signals:
    void finished(const WordInfo&);

public:
    ModelPipeline(){
        _labelingMode = QDir("labeled_data/").exists();
    }

    bool labelingMode(){return this->_labelingMode;}

    bool isReady(){
        return _isReady;
    }

    void run(QImage img){
        QString timestamp = QString::number(QDateTime::currentDateTime().toTime_t());

        if(labelingMode()){
            auto rootDir = QDir("labeled_data");
            rootDir.mkdir(timestamp);
            img.save(QString("labeled_data/%1/original.png").arg(timestamp));
        }

        QtConcurrent::run([=]{
            _isReady = false;

            QList<clock_t> timeList;

            timeList.append(clock());

            QImage cropped = detector.crop(img);

            if(cropped.isNull()){
                emit finished(WordInfo("(No word)"));
                _isReady = true;
                return;
            }

            if(labelingMode()){
                cropped.save(QString("labeled_data/%1/cropped.png").arg(timestamp));
            }

            timeList.append(clock());

            QString word = recognizer.predict(cropped.bits());

            if(labelingMode()){
                QFile file(QString("labeled_data/%1/label.txt").arg(timestamp));
                file.open(QIODevice::WriteOnly);
                file.write(word.toLatin1());
                file.close();
            }

            if(word.size() >= 8)
                word = corrector.correct(word);

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
