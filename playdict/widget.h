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
#include <QPropertyAnimation>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ui_widget.h"

#include "bingdict.h"
#include "recognizer.h"

#include "oescreenshot.h"
#include "qhook.h"
#include "uidefbar.h"

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
    QList<clock_t> timeList;

    QList<UiDefinitionBar*> bars;
    void updateUi(const WordInfo&);

    Recognizer recognizer;

    BingDict bingDict;

    QPoint mouseStartPoint, windowTopLeftPoint;
    QRect targetRect;

    QPoint targetPoint(){
        QRect rect = targetRect;

        if(rect.width() < 2 || rect.height() < 2)
            return pos();

        auto desktopSize = QApplication::desktop()->size();
        bool leftTag = rect.bottomRight().x() < desktopSize.width()-width();
        bool upTag = rect.bottomRight().y() < desktopSize.height()*0.92-height();
        if( leftTag &&  upTag) return rect.bottomRight();
        if( leftTag && !upTag) return rect.topRight()-QPoint(0,height());
        if(!leftTag &&  upTag) return rect.bottomLeft()-QPoint(width(),0);
        if(!leftTag && !upTag) return rect.topLeft()-QPoint(width(),height());
        return pos();
    }

    int renderPointX(){
        return 15;
    }

    int renderPointY(){
        return ui->pronBar->pos().y() + ui->pronBar->height() + 15;
    }

    int bottomMargin(){
        return 5;
    }

signals:
    void initialized();

private slots:
    void RegisterShortcuts(){
        hotkeys.append( new QHotkey(QKeySequence("F1"), true, this) );
        hotkeys.append( new QHotkey(QKeySequence("F2"), true, this) );
        hotkeys.append( new QHotkey(QKeySequence("F3"), true, this) );
        connect(hotkeys[0], SIGNAL(activated()), this, SLOT(screenShot()));
        connect(hotkeys[1], SIGNAL(activated()), this, SLOT(toggleVisible()));
        connect(hotkeys[2], SIGNAL(activated()), this, SLOT(close()));

        connect(QHook::Instance(), &QHook::mousePressed, [&](QHookMouseEvent *e){
            if(e->button()==QHookMouseEvent::MiddleButton)
                screenShot();
        });
    }

public slots:
    bool screenShot();
    void toggleVisible(){ setVisible(!isVisible()); }
    void onRecognizeFinished(QString word);
    void onQueryFinished(const WordInfo &wi);
};

#endif // WIDGET_H
