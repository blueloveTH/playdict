#include "oescreenshot.h"

OEScreenshot * OEScreenshot::self_ = nullptr;

OEScreenshot::OEScreenshot(QWidget *parent) : QWidget(parent){
    desktopRect = QRect(QApplication::desktop()->pos(),
                QApplication::desktop()->size());

    initGlobalScreen();
    setGeometry(desktopRect);

    ScreenUtil::setWindowFlags(this);

    setMouseTracking(true);
    show();
}

OEScreenshot::~OEScreenshot(void) {
}

OEScreenshot *OEScreenshot::Instance(void) {
    if (hasInstance()) delInstance();
    self_ = new OEScreenshot;
    return self_;
}

void OEScreenshot::initGlobalScreen(void) {
    QScreen *screen = QGuiApplication::primaryScreen();
    originPainting_.reset(new QPixmap(screen->grabWindow(0, desktopRect.x(),
                        desktopRect.y(), desktopRect.width(),
                        desktopRect.height())));
    /// Create a dark background
    QPixmap dim_pix(desktopRect.width(), desktopRect.height());
    dim_pix.fill((QColor(0, 0, 0, 24)));

    backgroundScreen_.reset(new QPixmap(*originPainting_));
    QPainter(backgroundScreen_.get()).drawPixmap(0, 0, dim_pix);
}

void OEScreenshot::createScreen(const QPoint &pos) {
    screenTool_.reset(new OEScreen(originPainting_, pos, this));
}

void OEScreenshot::destroyScreen() {
    if (screenTool_.get() != nullptr) {
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
        e->accept();
        return ;
    }
}

void OEScreenshot::mouseReleaseEvent(QMouseEvent *e) {
    if (isLeftPressed_ == true
             && e->button() == Qt::LeftButton) {
        isLeftPressed_ = false;

        QPixmap map = screenTool_->saveScreen();
        QRect rect = screenTool_->currentRect();

        destroyScreen();
        close();

        e->accept();
        emit finished(map, rect);
        delInstance();
    }
}

void OEScreenshot::mouseMoveEvent(QMouseEvent *e) {
    if (isLeftPressed_){
        if(screenTool_ != nullptr){
            screenTool_->onMouseChange(e->x(), e->y());
            e->accept();
        }
        update();
    }
}

void OEScreenshot::paintEvent(QPaintEvent *) {
    QPainter(this).drawPixmap(0,0,desktopRect.width(),
            desktopRect.height(), *backgroundScreen_);
}
