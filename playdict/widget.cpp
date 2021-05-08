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
}

bool Widget::screenShot()
{
    if(!recognizer.isReady() || !bingDict.isReady())
        return false;
    timeList.clear();
    auto o = OEScreenshot::Instance();
    connect(o, &OEScreenshot::finished, &recognizer, &Recognizer::exec);
    connect(o, &OEScreenshot::finished, [=]{
        ui->titleBar->setText("(Running...)");
        setVisible(true);
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
}

void Widget::updateUi(const WordInfo &wi){
    ui->titleBar->setText(wi.word);
    ui->pronBar->setText(wi.pronResult());

    while(!bars.empty()){
        delete bars.back();
        bars.pop_back();
    }

    int x = 20;
    int y = 108;

    if(ui->pronBar->text().isEmpty())
        y = ui->pronBar->pos().y() + 5;

    for(const auto &def : qAsConst(wi.definition)){
        auto bar = new UiDefinitionBar(this, x, y, def);
        y += bar->height() + 10;
        bars.append(bar);
    }

    setFixedHeight(y+20);

    update();
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
