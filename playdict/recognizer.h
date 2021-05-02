#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <QObject>
#include <QProcess>

class Recognizer : public QObject
{
    Q_OBJECT

    QProcess *c2tProcess;
    bool _isReady = true;

public:
    explicit Recognizer(QObject *parent = nullptr);
    bool isReady(){ return _isReady; }

signals:
    void finished(QString word, int code);

public slots:
    void exec();

private slots:
    void onCapture2TextFinished(int code);
};

#endif // RECOGNIZER_H
