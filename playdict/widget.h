#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QDebug>
#include <QBuffer>
#include <QObject>
#include <QSystemTrayIcon>
#include <QJsonDocument>
#include <QFileInfo>
#include <QDir>
#include <QMenu>

#include <QHotkey>
#include <QPropertyAnimation>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "ui_widget.h"
#include "oescreenshot.h"
#include "qhook.h"
#include "uidefbar.h"
#include "modelpipeline.h"

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QApplication* app, QWidget *parent = nullptr);
    ~Widget();

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseMoveEvent(QMouseEvent *);
    virtual void closeEvent(QCloseEvent *);

private:
    Ui::Widget *ui = nullptr;
    QList<QHotkey*> hotkeys;
    QSystemTrayIcon *trayIcon;
    QList<UiDefinitionBar*> bars;
    float scalingFactor;
    QApplication *app;

    QPoint mouseStartPoint, windowTopLeftPoint;
    QRect targetRect;

    QPoint targetPoint();
    void updateUi(const WordInfo&);
    int renderPointX(){ return 15;}
    int renderPointY(){ return ui->pronBar->pos().y() + ui->pronBar->height() + 15; }
    int bottomMargin(){ return 5;}
    int spacing(){ return 10;}

    ModelPipeline pipeline;

signals:
    void initialized();

private slots:
    void RegisterShortcuts();
    bool screenShot();
    void toggleVisible(){ setVisible(!isVisible()); }
    void onPipelineFinished(const WordInfo &wi);
};

#endif // WIDGET_H
