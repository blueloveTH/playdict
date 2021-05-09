#ifndef BINGDICT_H
#define BINGDICT_H

#include <QString>
#include <QList>
#include <QDomDocument>
#include <QDomNode>
#include "httplib.h"
#include "wordinfo.h"

class BingDict
{
    QString current_query;

    QList<QStringList> findAll(const QString& pattern_str, const QString&, int);
    QList<int> findAllIndex(const QString& pattern_str, const QString&, int);
    QString subStringDiv(QString text, int startPos);

    httplib::Client *client;

    WordInfo parse(QByteArray);

public:
    BingDict();
    ~BingDict() {}
    WordInfo query(QString q);
};

#endif // BINGDICT_H
