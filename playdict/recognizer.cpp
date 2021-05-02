#include "recognizer.h"

Recognizer::Recognizer(QObject *parent) : QObject(parent)
{

}

void Recognizer::exec(){
    if(!_isReady)
        return;
    _isReady = false;
    c2tProcess = new QProcess(this);
    connect(c2tProcess, SIGNAL(finished(int)), this, SLOT(onCapture2TextFinished(int)));
    QStringList args{"-i", "tmp.png"};
    c2tProcess->start("./Capture2Text/Capture2Text_CLI.exe", args);
}


void Recognizer::onCapture2TextFinished(int code){
    if(code != 0)
        emit finished("", code);
    QString word = c2tProcess->readAllStandardOutput();
    c2tProcess->deleteLater();
    emit finished(word, code);
    _isReady = true;
}
