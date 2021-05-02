#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

#include <QDebug>
#include <QBuffer>
#include <QObject>

#include <QtWebView>
#include "jsondict.cpp"
#include "recognizer.h"

#include "ui_widget.h"
#include "oescreenshot.h"
#include "qxt/qxtglobalshortcut.h"
#include <QClipboard>
#include <QMimeData>

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
    virtual void mouseMoveEvent(QMouseEvent *e);

    virtual void closeEvent(QCloseEvent* event){ exit(0); }

private:
    Ui::Widget *ui;
    QJsonDocument config;
    Recognizer recognizer;
    JsonDict jsonDict;

    QPoint mouseStartPoint, windowTopLeftPoint;

public slots:
    void screenShot();
    void toggleVisible(){ setVisible(!isVisible()); }
    void onRecognizeFinished(QString word, int code);
};

#endif // WIDGET_H
