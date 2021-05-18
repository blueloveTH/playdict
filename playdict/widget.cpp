#include "widget.h"

Widget::Widget(QApplication* app, QWidget* parent) :
    QWidget(parent),
    ui(new Ui::Widget), app(app)
{
    ui->setupUi(this);
    /// Set windows
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::WindowStaysOnTopHint;
    flags |= Qt::FramelessWindowHint;
    setWindowFlags(flags);

    /// No focus
    HWND wid = (HWND)(this->winId());
        SetWindowLong(wid, GWL_EXSTYLE, GetWindowLong(wid, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_COMPOSITED);

    qRegisterMetaType<WordInfo>("WordInfo");
    connect(&pipeline, &ModelPipeline::finished, this, &Widget::onPipelineFinished);

    trayIcon = new QSystemTrayIcon(QIcon(QPixmap(32, 32)), this);
    trayIcon->show();

    connect(this, SIGNAL(initialized()), this, SLOT(RegisterShortcuts()));
    QtConcurrent::run([=]{Sleep(300);emit initialized();});

    updateUi(WordInfo::helpWord());

    labelingMode = QDir("labeled_data/").exists();

    if(labelingMode){
        QString text = "Labeling Mode (cnt=%1)";
        text = text.arg(QDir("labeled_data/").count());
        ui->pronBar->setText(text);
    }

}

bool Widget::screenShot()
{
    if(!pipeline.isReady()) return false;

    if(OEScreenshot::hasInstance()){
        OEScreenshot::delInstance();
        return false;
    }

    auto o = OEScreenshot::Instance();
    connect(o, &OEScreenshot::finished, [=](QPixmap map, QRect rect){
        targetRect = rect;

        ui->titleBar->setText("(Running...)");
        ui->pronBar->setText("");

        while(!bars.empty()){
            delete bars.back();
            bars.pop_back();
        }

        setFixedHeight(renderPointY()+bottomMargin());
        update();

        /********************************/
        pipeline.run(map.toImage());
    });
    return true;
}

void Widget::updateUi(const WordInfo& wi){
    ui->titleBar->setText(wi.word);

    int font_size = font().pixelSize();
    QString css = QString("font-size: %1px;");
    ui->pronBar->setStyleSheet(css.arg(font_size));
    if(wi.hasResult())
        ui->pronBar->setText(wi.pronResult());
    else
        ui->pronBar->setText("(No result)");
    ui->pronBar->adjustSize();

    int x = renderPointX();
    while(ui->pronBar->width() > width()-x*2){
        font_size--;
        ui->pronBar->setStyleSheet(css.arg(font_size));
        ui->pronBar->adjustSize();
    }

    int y = renderPointY();

    if(ui->pronBar->text().isEmpty())
        y = ui->pronBar->pos().y() + spacing();

    for(const auto &def : qAsConst(wi.definition)){
        auto bar = new UiDefinitionBar(this, x, y, def);
        y += bar->height() + spacing();
        bars.append(bar);
    }

    setFixedHeight(y+bottomMargin());
}

void Widget::onPipelineFinished(const WordInfo& wi){
    setVisible(false);

    updateUi(wi);
    update();
    move(targetPoint());

    setVisible(true);

    if(labelingMode){
        QString timestamp = QString::number(QDateTime::currentDateTime().toTime_t());
        QString filename = QString("labeled_data/%1_%2.png").arg(wi.word).arg(timestamp);
        pipeline.currImg().save(filename);
    }

    /*QPropertyAnimation *animation = new QPropertyAnimation(this, "pos");
    animation->setDuration(1);
    animation->setStartValue(pos());
    animation->setEndValue(targetPoint());
    animation->setEasingCurve(QEasingCurve::OutQuad);
    animation->start(QAbstractAnimation::DeleteWhenStopped);*/
}

QPoint Widget::targetPoint(){
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

void Widget::RegisterShortcuts(){
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

void Widget::closeEvent(QCloseEvent *e){
    for(int i=0; i<hotkeys.count(); i++)
        hotkeys[i]->setRegistered(false);
    delete trayIcon;
    e->accept();
    app->quit();
}

Widget::~Widget()
{
    delete ui;
}

////////////////

void Widget::mousePressEvent(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton)
    {
        mouseStartPoint = event->globalPos();
        windowTopLeftPoint = this->frameGeometry().topLeft();
        event->accept();
    }
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->buttons() & Qt::LeftButton)
    {
        QPoint diff = event->globalPos() - mouseStartPoint;
        this->move(windowTopLeftPoint + diff);
        event->accept();
    }
}
