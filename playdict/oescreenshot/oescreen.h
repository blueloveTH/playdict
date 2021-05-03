#ifndef OESCREEN_H
#define OESCREEN_H

#include <QPainter>
#include <QScreen>
#include <QWidget>
#include <QClipboard>
#include <QApplication>

#include <windows.h>

class OEScreen : public QWidget {
    Q_OBJECT
protected:

    const int PADDING_ = 6;

public:

    explicit OEScreen(std::shared_ptr<QPixmap> originPainting, QPoint pos, QWidget *parent = 0);

    ~OEScreen() { }

protected:
    virtual void paintEvent(QPaintEvent *);

public slots:
    void onMouseChange(int x,int y);
    void onSaveScreen(void);

private:
    QPoint          originPoint_;
    std::shared_ptr<QPixmap> originPainting_;
    QRect           currentRect_;
};
#endif // OESCREEN_H
