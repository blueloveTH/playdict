#ifndef DETECTOR_H
#define DETECTOR_H

#include <QObject>

class Detector : public QObject
{
    Q_OBJECT
public:
    explicit Detector(QObject *parent = nullptr);

    void detectNearestBox(const QImage& img){

    }
signals:

};

#endif // DETECTOR_H
