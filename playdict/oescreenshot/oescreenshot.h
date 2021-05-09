#ifndef OESCREENSHOT_H
#define OESCREENSHOT_H

#include <QMouseEvent>
#include <QPainter>
#include <QScreen>
#include <QMutex>
#include <QPen>
#include <QDebug>
#include <QRect>
#include <QWidget>
#include <QDesktopWidget>

#include "oescreen.h"

class OEScreen;

class OEScreenshot : public QWidget {
    Q_OBJECT

signals:
    void finished(QPixmap, QRect);
public:
    explicit OEScreenshot(QWidget *parent = 0);
    ~OEScreenshot(void);

    static OEScreenshot *Instance(void);

    bool static hasInstance(){
        return self_;
    }

    void static delInstance(){
        delete self_;
        self_ = nullptr;
    }

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void paintEvent(QPaintEvent *);

private:
    void initGlobalScreen(void);
    void createScreen(const QPoint &pos);
    void destroyScreen(void);

private:
    bool                        isLeftPressed_ = false;
    QRect desktopRect;
    std::shared_ptr<QPixmap>    backgroundScreen_ = nullptr;
    std::shared_ptr<QPixmap>    originPainting_ = nullptr;
    std::shared_ptr<OEScreen>   screenTool_ = nullptr;
    static OEScreenshot         *self_;
};

#endif /// OESCREENSHOT_H
