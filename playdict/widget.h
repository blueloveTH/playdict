#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QDebug>
#include <QBuffer>
#include <QObject>
#include <QSystemTrayIcon>
#include <QtConcurrent/QtConcurrentRun>

#include <QHotkey>
#include <windows.h>

#include "jsondict.cpp"
#include "recognizer.h"

#include "ui_widget.h"
#include "oescreenshot.h"

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
    virtual void mouseMoveEvent(QMouseEvent *);
    virtual void closeEvent(QCloseEvent *);

private:
    Ui::Widget *ui = nullptr;
    QList<QHotkey*> hotkeys;
    QSystemTrayIcon *trayIcon = nullptr;

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
