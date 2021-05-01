#include "oescreenshot.h"

#include <QDesktopWidget>
#include <QApplication>
#include <QMouseEvent>
#include <QFileDialog>
#include <QClipboard>
#include <QDateTime>
#include <QPainter>
#include <QScreen>
#include <QCursor>
#include <QMutex>
#include <QMenu>
#include <QPen>
#ifndef QT_NO_DEBUG
#include <QDebug>
#endif

#include <windows.h>

#include "oecommonhelper.h"


OEScreenshot * OEScreenshot::self_ = nullptr;

OEScreenshot::OEScreenshot(QWidget *parent) : QWidget(parent),
    isLeftPressed_ (false), backgroundScreen_(nullptr),
    originPainting_(nullptr), screenTool_(nullptr) {
    /// 截取屏幕信息
    initGlobalScreen();
    /// 窗口与显示屏对齐
    setGeometry(getScreenRect());

    /// 窗口置顶
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::WindowStaysOnTopHint;
    flags |= Qt::FramelessWindowHint;
    setWindowFlags(flags);

    HWND wid = (HWND)(this->winId());
        SetWindowLong(wid, GWL_EXSTYLE, GetWindowLong(wid, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_COMPOSITED);

    /// 开启鼠标实时追踪
    setMouseTracking(true);
    show();
}

OEScreenshot::~OEScreenshot(void) {
}

OEScreenshot *OEScreenshot::Instance(void) {
    if (self_) destroy();
    self_ = new OEScreenshot;
    return self_;
}

void OEScreenshot::destroy(void) {
    if (self_) {
        delete self_;
        self_ = nullptr;
    }
}

/**
 * 功能：获得当前屏幕的大小
 */
const QRect &OEScreenshot::getScreenRect(void) {
    if (!desktopRect_.isEmpty()) {
        return desktopRect_;
    }
    /// 兼容多个屏幕的问题
    desktopRect_ = QRect(QApplication::desktop()->pos(),
          QApplication::desktop()->size());
    return desktopRect_;
}

void OEScreenshot::initGlobalScreen(void) {
    /// 获得屏幕原画
    /// 截取当前桌面，作为截屏的背景图
    QScreen *screen = QGuiApplication::primaryScreen();
    const QRect& temp_rect = getScreenRect();
    originPainting_.reset(new QPixmap(screen->grabWindow(0, temp_rect.x(),
                        temp_rect.y(), temp_rect.width(),
                        temp_rect.height())));

    std::shared_ptr<QPixmap> temp_screen = originPainting_;

    /// 制作暗色屏幕背景
    QPixmap temp_dim_pix(temp_screen->width(), temp_screen->height());
    temp_dim_pix.fill((QColor(0, 0, 0, 16)));
    backgroundScreen_.reset(new QPixmap(*temp_screen));
    QPainter p(backgroundScreen_.get());
    p.drawPixmap(0, 0, temp_dim_pix);
}

void OEScreenshot::createScreen(const QPoint &pos) {
    /// 创建截图器
    screenTool_.reset(new OEScreen(originPainting_, pos, this));
}

void OEScreenshot::destroyScreen() {
    if (screenTool_.get() != nullptr) {
        /// 清理工具
        screenTool_.reset();
        screenTool_ = nullptr;
        isLeftPressed_ = false;
        update();
        return;
    }
}

void OEScreenshot::mousePressEvent(QMouseEvent *e) {
    if (e->button() == Qt::LeftButton) {
        createScreen(e->pos());
        isLeftPressed_ = true;
        return ;
    }
}

void OEScreenshot::mouseReleaseEvent(QMouseEvent *e) {
    if (isLeftPressed_ == true
             && e->button() == Qt::LeftButton) {
        isLeftPressed_ = false;

        screenTool_->onSaveScreen();
        emit onScreenshot();

        destroyScreen();
        close();
    }
}

void OEScreenshot::mouseMoveEvent(QMouseEvent *e) {
    if (isLeftPressed_){
        if(screenTool_ != nullptr)
            screenTool_->onMouseChange(e->x(), e->y());
        update();
    }
}

void OEScreenshot::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    /// 画全屏图
    painter.drawPixmap(0,0,desktopRect_.width(),
            desktopRect_.height(), *backgroundScreen_);
}

///////////////////////////////////////////////////////////


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
    /// 把图片放入剪切板
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
    /// 改变大小后更新父窗口，防止父窗口未及时刷新而导致的问题
    parentWidget()->update();
}
