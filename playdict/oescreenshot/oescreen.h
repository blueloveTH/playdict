#ifndef OESCREEN_H
#define OESCREEN_H

#include <QPainter>
#include <QScreen>
#include <QWidget>
#include <QApplication>

#include <windows.h>

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
        QColor c0 = originPaintingImage_.pixelColor(currentRect_.topLeft());
        QColor c1 = originPaintingImage_.pixelColor(currentRect_.topRight());
        QColor c2 = originPaintingImage_.pixelColor(currentRect_.bottomLeft());
        QColor c3 = originPaintingImage_.pixelColor(currentRect_.bottomRight());
        return (c0.lightnessF()+c1.lightnessF()+c2.lightnessF()+c3.lightnessF()) / 4;
    }
};
#endif // OESCREEN_H
