#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <QObject>
#include <QProcess>
#include <QtDebug>
#include <QPixmap>
#include <QImage>
#include <QtConcurrent/QtConcurrentRun>
#include "onnxruntime_cxx_api.h"

class Recognizer : public QObject
{
    Q_OBJECT

    QProcess *c2tProcess = nullptr;
    bool _isReady = true;

    Ort::Session *session;

    QString model_predict(const QPixmap &map);

public:
    explicit Recognizer(QObject *parent = nullptr);
    bool isReady(){ return _isReady; }

signals:
    void finished(QString word);

public slots:
    void exec(QPixmap map);
};

#endif // RECOGNIZER_H
