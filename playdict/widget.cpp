#include "widget.h"

Widget::Widget(QApplication* app, QWidget* parent) :
    QWidget(parent),
    ui(new Ui::Widget), app(app)
{
    /// Global stylesheet
    QFile qssFile(":/ui/qss/res/qss/vscode.qss");
    qssFile.open(QIODevice::ReadOnly);
    app->setStyleSheet(qssFile.readAll());
    qssFile.close();

    ui->setupUi(this);

    ScreenUtil::setWindowFlags(this);

    qRegisterMetaType<WordInfo>("WordInfo");
    connect(&pipeline, &ModelPipeline::finished, this, &Widget::onPipelineFinished);

    /// Tray icon
    trayIcon = new QSystemTrayIcon(QIcon(QPixmap(":/ui/res/ico.png").scaled(32,32)), this);
    QMenu *trayMenu = new QMenu(this);
    trayMenu->addAction("Exit", [=]{app->exit();});
    trayIcon->setContextMenu(trayMenu);
    connect(trayIcon, &QSystemTrayIcon::activated, [=]{setVisible(true); trayMenu->activateWindow();});
    trayIcon->show();

    /// Shortcuts
    connect(this, SIGNAL(initialized()), this, SLOT(RegisterShortcuts()));
    QtConcurrent::run([=]{Sleep(300);emit initialized();});

    updateUi(WordInfo::helpWord());

    if(pipeline.labelingMode()){
        QString text = "(Labeling Mode: ENABLED)";
        ui->pronBar->setText(text);
        ui->pronBar->adjustSize();
    }

    /*QString css = styleSheet();
    auto re_1 = QRegExp("background:.*;"); re_1.setMinimal(true);
    auto re_2 = QRegExp("color:.*;"); re_1.setMinimal(true);
    css = css.replace(re_1, "background: rgb(255,255,255);");
    css = css.replace(re_2, "color: rgb(0,0,0);");
    setStyleSheet(css);*/
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
    hotkeys.append( new QHotkey(QKeySequence("F4"), true, this) );
    connect(hotkeys[0], SIGNAL(activated()), this, SLOT(screenShot()));
    connect(hotkeys[1], SIGNAL(activated()), this, SLOT(toggleVisible()));
    connect(hotkeys[2], SIGNAL(activated()), this, SLOT(close()));
    connect(ui->hideButton, SIGNAL(pressed()), this, SLOT(toggleVisible()));

    connect(QHook::Instance(), &QHook::mousePressed, [&](QHookMouseEvent *e){
        if(e->button()==QHookMouseEvent::MiddleButton)
            screenShot();
    });

    connect(hotkeys[3], &QHotkey::activated, [=]{
        HWND hwnd = GetForegroundWindow();
        if(hwnd == (HWND)this->winId()) return;

        WCHAR* text = new WCHAR[32];
        GetWindowText(hwnd, text, 32);

        bool is_vis = isVisible();
        setVisible(false);

        QString descrption = QString::fromWCharArray(text);
        descrption = "Convert window \"" + descrption + "\" to fullscreen borderless mode?";
        auto btn = QMessageBox::question(
                    this,
                    QString("Confirm"),
                    descrption
                    );
        if(btn == QMessageBox::StandardButton::Yes){
            SetWindowLong(hwnd, GWL_STYLE, GetWindowLong(hwnd, GWL_STYLE) & ~( WS_THICKFRAME | WS_BORDER | WS_DLGFRAME | WS_CAPTION | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU ) );
            QSize size = ScreenUtil::getDesktopWH();
            SetWindowPos(hwnd, NULL, 0, 0, size.width(), size.height(), SWP_SHOWWINDOW);
        }

        setVisible(is_vis);
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
