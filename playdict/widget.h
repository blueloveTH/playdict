#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>

#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>

#include <QDebug>
#include <QBuffer>
#include <QObject>

#include <QtWebView>
#include <QProcess>

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget();

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);

    virtual void closeEvent(QCloseEvent* event);

private:
    Ui::Widget *ui;
    QJsonDocument config;

    QNetworkReply *reply=0;
    QProcess* c2tProcess=0;

    QHash<QString, QString> hashMap;
    QJsonArray dicArray;

    QPoint mouseStartPoint, windowTopLeftPoint;
    bool _drag;

    void wordQueue(QString word);

public slots:
    void shot();
    void parse();
    void onRequestFinished();
    void toggleVisible();

    void onCapture2TextFinished(int code);
};

#endif // WIDGET_H
