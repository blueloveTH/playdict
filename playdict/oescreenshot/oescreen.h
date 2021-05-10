#ifndef OESCREEN_H
#define OESCREEN_H

#include <QPainter>
#include <QScreen>
#include <QWidget>
#include <QApplication>
#include <QtDebug>
#include "screenutil.h"

class OEScreen : public QWidget {
    Q_OBJECT
protected:

    const int PADDING_ = 6;

public:

    explicit OEScreen(std::shared_ptr<QPixmap>, QPoint, QWidget *parent = 0);

    ~OEScreen() { }

    QRect currentRect(){
        return currentRect_;
    }

    QRect scaledCurrentRect(){
        QRect rect = currentRect();
        float x = rect.x() * ScreenUtil::scalingFactor();
        float y = rect.y() * ScreenUtil::scalingFactor();
        float w = rect.width() * ScreenUtil::scalingFactor();
        float h = rect.height() * ScreenUtil::scalingFactor();
        return QRect(round(x), round(y), round(w), round(h));
    }

protected:
    virtual void paintEvent(QPaintEvent *);

public slots:
    void onMouseChange(int x,int y);
    QPixmap saveScreen(void);

private:
    QPoint          originPoint_;
    std::shared_ptr<QPixmap> originPainting_;
    QImage          originPaintingImage_;
    QRect           currentRect_;

    float bkgLightness(){
        QRect rect = scaledCurrentRect();
        QColor c0 = originPaintingImage_.pixelColor(rect.topLeft());
        QColor c1 = originPaintingImage_.pixelColor(rect.topRight());
        QColor c2 = originPaintingImage_.pixelColor(rect.bottomLeft());
        QColor c3 = originPaintingImage_.pixelColor(rect.bottomRight());
        return (c0.lightnessF()+c1.lightnessF()+c2.lightnessF()+c3.lightnessF()) / 4;
    }
};
#endif // OESCREEN_H
