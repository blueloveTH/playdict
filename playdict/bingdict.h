#ifndef BINGDICT_H
#define BINGDICT_H

#include <QString>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QList>

class BingDict : public QObject
{
    Q_OBJECT

    QNetworkReply *reply = nullptr;
    QList<QStringList> findAll(QRegExp&, const QString&);
    bool _isReady = true;
    QString current_query;

signals:
    void finished(QString);

public:
    BingDict() {connect(this, &BingDict::finished, [&]{_isReady=true;});}
    virtual ~BingDict() {}
    void query(QString q);

    bool isReady(){ return _isReady; }

private slots:
    void onRequestFinished();
};

#endif // BINGDICT_H
