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

    explicit OEScreen(std::shared_ptr<QPixmap>, QPoint, QColor, QWidget *parent = 0);

    ~OEScreen() { }

protected:
    virtual void paintEvent(QPaintEvent *);

public slots:
    void onMouseChange(int x,int y);
    QPixmap saveScreen(void);

private:
    QPoint          originPoint_;
    std::shared_ptr<QPixmap> originPainting_;
    QRect           currentRect_;
    QColor          borderColor;
};
#endif // OESCREEN_H
