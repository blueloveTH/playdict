#include "oescreen.h"

OEScreen::OEScreen(std::shared_ptr<QPixmap> originPainting, QPoint pos, QWidget *parent)
    : QWidget(parent), originPoint_(pos), originPainting_(originPainting) {

    HWND wid = (HWND)(this->winId());
        SetWindowLong(wid, GWL_EXSTYLE, GetWindowLong(wid, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_COMPOSITED);
}

void OEScreen::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    /// 绘制截屏编辑窗口
    painter.drawPixmap(QPoint(0,0),
       *originPainting_, currentRect_);

    /// 绘制边框线
    QPen pen(QColor(0,174,255), 4);
    painter.setPen(pen);
    painter.drawRect(rect());
}


void OEScreen::onSaveScreen(void) {
    QClipboard *board = QApplication::clipboard();
    QPixmap map = originPainting_->copy(currentRect_);
    board->setPixmap(map);
    map.save("tmp.png");
}

void OEScreen::onMouseChange(int x, int y) {
    show();
    if (x < 0 || y < 0) {
        return;
    }
    const int& rx = (x >= originPoint_.x()) ? originPoint_.x() : x;
    const int& ry = (y >= originPoint_.y()) ? originPoint_.y() : y;
    const int& rw = abs(x - originPoint_.x());
    const int& rh = abs(y - originPoint_.y());

    /// 改变大小
    currentRect_ = QRect(rx, ry, rw, rh);

    this->setGeometry(currentRect_);
    parentWidget()->update();
}

