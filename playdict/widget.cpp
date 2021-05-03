#include "widget.h"

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);

    /// Shortcuts
    hotkeys.append( new QHotkey(QKeySequence("F1"), true, this) );
    hotkeys.append( new QHotkey(QKeySequence("F2"), true, this) );
    hotkeys.append( new QHotkey(QKeySequence("F3"), true, this) );
    connect(hotkeys[0], SIGNAL(activated()), this, SLOT(screenShot()));
    connect(hotkeys[1], SIGNAL(activated()), this, SLOT(toggleVisible()));
    connect(hotkeys[2], SIGNAL(activated()), this, SLOT(close()));

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

    /// Load json dictionary
    QtConcurrent::run([&]{jsonDict.load("ecdict.json");});

    /// Setup recognizer
    connect(&recognizer, SIGNAL(finished(QString, int)), this, SLOT(onRecognizeFinished(QString, int)));

    trayIcon = new QSystemTrayIcon(QIcon(QPixmap(32, 32)), this);
    trayIcon->show();
}

void Widget::screenShot()
{
    if(!recognizer.isReady())
        return;
    auto o = OEScreenshot::Instance();
    connect(o, SIGNAL(finished()), &recognizer, SLOT(exec()));
    connect(o, &OEScreenshot::finished, [=]{
        ui->textEdit->setText("(Running...)");
        setVisible(true);
        update();
    });
}


void Widget::onRecognizeFinished(QString word, int code){
    QFile::remove("tmp.png");
    ui->textEdit->setText(jsonDict.query(word));
    update();
}


void Widget::closeEvent(QCloseEvent *e){
    for(int i=0; i<hotkeys.count(); i++)
        hotkeys[i]->setRegistered(false);
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
    }
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    if(event->buttons() & Qt::LeftButton)
    {
        QPoint diff = event->globalPos() - mouseStartPoint;
        this->move(windowTopLeftPoint + diff);
    }
}
