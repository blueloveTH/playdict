#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include <QString>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

class Translator: public QObject
{
    Q_OBJECT

signals:
    void onFinished(QString q);
public:
    Translator(const QString& queue);
private:
    void onRequestFinished();
    QNetworkReply* reply;
};

#endif // TRANSLATOR_H
