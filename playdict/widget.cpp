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

    /// Load config file
    QFile cfgFile("config.json");
    cfgFile.open(QIODevice::ReadOnly);
    config = QJsonDocument::fromJson(cfgFile.readAll());
    cfgFile.close();

    connect(&bingDict, &BingDict::finished, this, &Widget::onQueryFinished);
    connect(&recognizer, SIGNAL(finished(QString, int)), this, SLOT(onRecognizeFinished(QString, int)));

    trayIcon = new QSystemTrayIcon(QIcon(QPixmap(32, 32)), this);
    trayIcon->show();

    RegisterShortcuts();
}

bool Widget::screenShot()
{
    if(!recognizer.isReady() || !bingDict.isReady())
        return false;
    auto o = OEScreenshot::Instance();
    connect(o, SIGNAL(finished()), &recognizer, SLOT(exec()));
    connect(o, &OEScreenshot::finished, [=]{
        ui->textEdit->setText("(Running...)");
        setVisible(true);
        update();
    });
    return true;
}


void Widget::onRecognizeFinished(QString word, int code){
    QFile::remove("tmp.png");
    bingDict.query(word);
}

void Widget::onQueryFinished(QString result){
    ui->textEdit->setText(result);
    ui->textEdit->setFixedHeight(ui->textEdit->document()->size().height());
    setFixedHeight(ui->textEdit->height()+10);
    update();
}


void Widget::closeEvent(QCloseEvent *e){
    for(int i=0; i<hotkeys.count(); i++)
        hotkeys[i]->setRegistered(false);
    PythonEnv::Finalize();
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
