#ifndef BINGDICT_H
#define BINGDICT_H

#include <QString>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QList>
#include <QDomDocument>
#include <QDomNode>

class BingDict : public QObject
{
    Q_OBJECT

    bool _isReady = true;
    QString current_query;
    QNetworkAccessManager manager;

    QList<QStringList> findAll(const QString& pattern_str, const QString&, int);
    QList<int> findAllIndex(const QString& pattern_str, const QString&, int);
    QString subStringDiv(QString text, int startPos);

signals:
    void finished(QString);

public:
    BingDict() {connect(this, &BingDict::finished, [&]{_isReady=true;});}
    virtual ~BingDict() {}
    void query(QString q);

    bool isReady(){ return _isReady; }

private slots:
    void onReply(QNetworkReply*);
};

#endif // BINGDICT_H
