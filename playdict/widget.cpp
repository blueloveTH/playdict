#include "widget.h"

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
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

    connect(&bingDict, &BingDict::finished, this, &Widget::onQueryFinished);
    connect(&recognizer, SIGNAL(finished(QString)), this, SLOT(onRecognizeFinished(QString)));

    trayIcon = new QSystemTrayIcon(QIcon(QPixmap(32, 32)), this);
    trayIcon->show();

    connect(this, SIGNAL(initialized()), this, SLOT(RegisterShortcuts()));
    QtConcurrent::run([=]{Sleep(300);emit initialized();});

    updateUi(WordInfo::helpWord());
}

bool Widget::screenShot()
{
    if(!recognizer.isReady() || !bingDict.isReady())
        return false;
    timeList.clear();
    auto o = OEScreenshot::Instance();
    connect(o, &OEScreenshot::finished, &recognizer, &Recognizer::exec);
    connect(o, &OEScreenshot::finished, [=](QPixmap _, QRect rect){
        targetRect = rect;

        ui->titleBar->setText("(Running...)");
        ui->pronBar->setText("");

        while(!bars.empty()){
            delete bars.back();
            bars.pop_back();
        }

        setFixedHeight(108+20);
        update();
        timeList.append(clock());
    });
    return true;
}


void Widget::onRecognizeFinished(QString word){
    timeList.append(clock());
    bingDict.query(word);
}

void Widget::onQueryFinished(const WordInfo& wi){
    timeList.append(clock());
    clock_t cost_0 = timeList[1] - timeList[0];
    clock_t cost_1 = timeList[2] - timeList[1];
    QString extraStr = "";
    extraStr += QString("\nRe: ") + QString::number(cost_0) + "ms";
    extraStr += QString("\tSe: ") + QString::number(cost_1) + "ms";

    updateUi(wi);
    move(targetPoint());
    update();
    setVisible(true);

    /*
        QPropertyAnimation *animation = new QPropertyAnimation(this, "pos");
        animation->setDuration(250);
        animation->setStartValue(pos());
        animation->setEndValue(targetPoint());
        animation->setEasingCurve(QEasingCurve::OutQuad);
        animation->start(QAbstractAnimation::DeleteWhenStopped);
    */
}

void Widget::updateUi(const WordInfo &wi){
    ui->titleBar->setText(wi.word);

    int font_size = font().pixelSize();
    QString css = QString("font-size: %1px;");
    ui->pronBar->setStyleSheet(css.arg(font_size));
    if(wi.hasResult())
        ui->pronBar->setText(wi.pronResult());
    else
        ui->pronBar->setText("(No result)");
    ui->pronBar->adjustSize();

    int x = 15;

    while(ui->pronBar->width() > width()-x*2){
        font_size--;
        ui->pronBar->setStyleSheet(css.arg(font_size));
        ui->pronBar->adjustSize();
    }


    int y = ui->pronBar->pos().y() + ui->pronBar->height() + 15;

    if(ui->pronBar->text().isEmpty())
        y = ui->pronBar->pos().y() + 10;

    for(const auto &def : qAsConst(wi.definition)){
        auto bar = new UiDefinitionBar(this, x, y, def);
        y += bar->height() + 10;
        bars.append(bar);
    }

    setFixedHeight(y+5);
}


void Widget::closeEvent(QCloseEvent *e){
    for(int i=0; i<hotkeys.count(); i++)
        hotkeys[i]->setRegistered(false);
    delete trayIcon;
    e->accept();
    exit(0);
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
