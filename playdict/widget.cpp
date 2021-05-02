#include "widget.h"

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);

    /// Shortcuts
    QxtGlobalShortcut* shortcut_1 = new QxtGlobalShortcut(QKeySequence("F1"), this);
    QxtGlobalShortcut* shortcut_2 = new QxtGlobalShortcut(QKeySequence("F2"), this);
    QxtGlobalShortcut* shortcut_3 = new QxtGlobalShortcut(QKeySequence("F3"), this);
    connect(shortcut_1, SIGNAL(activated()), this, SLOT(screenShot()));
    connect(shortcut_2, SIGNAL(activated()), this, SLOT(toggleVisible()));
    connect(shortcut_3, SIGNAL(activated()), this, SLOT(close()));

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
    jsonDict.load("ecdict.json");

    /// Setup recognizer
    connect(&recognizer, SIGNAL(finished(QString, int)), this, SLOT(onRecognizeFinished(QString, int)));
}

void Widget::screenShot()
{
    if(!recognizer.isReady())
        return;
    auto o = OEScreenshot::Instance();
    connect(o, SIGNAL(onScreenshot()), &recognizer, SLOT(exec()));
    connect(o, &OEScreenshot::onScreenshot, [=]{
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
