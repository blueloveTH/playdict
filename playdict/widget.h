#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QDebug>
#include <QBuffer>
#include <QObject>
#include <QSystemTrayIcon>
#include <QtConcurrent/QtConcurrentRun>
#include <QJsonDocument>

#include <QHotkey>
#include <windows.h>

#include "bingdict.h"
#include "recognizer.h"

#include "ui_widget.h"
#include "oescreenshot.h"
#include "qhook.h"

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

    BingDict bingDict;

    QPoint mouseStartPoint, windowTopLeftPoint;

public slots:
    bool screenShot();
    void toggleVisible(){ setVisible(!isVisible()); }
    void onRecognizeFinished(QString word, int code);
    void onQueryFinished(QString result);
};

#endif // WIDGET_H
