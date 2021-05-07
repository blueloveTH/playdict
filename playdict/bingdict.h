#ifndef BINGDICT_H
#define BINGDICT_H

#include <QString>
#include <QList>
#include <QDomDocument>
#include <QDomNode>
#include <QtConcurrent/QtConcurrentRun>
#include "httplib.h"

class BingDict : public QObject
{
    Q_OBJECT

    bool _isReady = true;
    QString current_query;

    QList<QStringList> findAll(const QString& pattern_str, const QString&, int);
    QList<int> findAllIndex(const QString& pattern_str, const QString&, int);
    QString subStringDiv(QString text, int startPos);

    //PyObject *pyQueryFunc;

    httplib::Client *client;

signals:
    void finished(QString);

public:
    BingDict();
    ~BingDict() {}
    void query(QString q);

    bool isReady(){ return _isReady; }

private slots:
    void onReply(QByteArray);
};

#endif // BINGDICT_H
