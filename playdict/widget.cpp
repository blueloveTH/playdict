#include "widget.h"
#include "ui_widget.h"
#include "oescreenshot.h"
#include "qxt/qxtglobalshortcut.h"
#include <QClipboard>
#include <QMimeData>
#include <QDebug>
#include <QBuffer>
#include <QObject>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>

#include <QtWebView>

#include "translator.h"

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);

    if(QFile::exists("tmp.png"))
        QFile::remove("tmp.png");

    QxtGlobalShortcut* shortcut = new QxtGlobalShortcut(QKeySequence("F1"), this);
    QxtGlobalShortcut* shortcut_2 = new QxtGlobalShortcut(QKeySequence("F2"), this);
    QxtGlobalShortcut* shortcut_3 = new QxtGlobalShortcut(QKeySequence("F3"), this);
    connect(shortcut, SIGNAL(activated()), this, SLOT(shot()));
    connect(shortcut_2, SIGNAL(activated()), this, SLOT(toggleVisible()));
    connect(shortcut_3, SIGNAL(activated()), this, SLOT(close()));

    /// 窗口置顶
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::WindowStaysOnTopHint;
    flags |= Qt::FramelessWindowHint;
    setWindowFlags(flags);

    HWND wid = (HWND)(this->winId());
        SetWindowLong(wid, GWL_EXSTYLE, GetWindowLong(wid, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_COMPOSITED);

    // 加载配置文件
    QFile file_0("config.json");
    file_0.open(QIODevice::ReadOnly);
    QJsonDocument config = QJsonDocument::fromJson(file_0.readAll());
    access_token = config["access_token"].toString();
    file_0.close();

    // 加载字典
    QFile file("ecdict.json");
    file.open(QIODevice::ReadOnly);

    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();

    dicArray = doc["data"].toArray();
    for(auto i=dicArray.constBegin(); i!=dicArray.constEnd(); i++){
        auto a = i->toArray();
        hashMap[a[0].toString()] = a[1].toString();
    }

    qDebug() << "Json dict loaded." << hashMap.count();
}

void Widget::closeEvent(QCloseEvent* event){
    exit(0);
}

void Widget::toggleVisible()
{
    setVisible(!isVisible());
}

void Widget::shot()
{
    auto o = OEScreenshot::Instance();
    connect(o, SIGNAL(onScreenshot()), this, SLOT(parse()));
}

/*
void Widget::parse()
{
    QByteArray ba;
    QBuffer buf(&ba);
    QImage("tmp.png").save(&buf, "png");

    QNetworkRequest request;
    request.setUrl(QUrl("https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token=" + access_token));
    request.setRawHeader("content-type", "application/x-www-form-urlencoded");

    QString data = "image=" + QString(ba.toBase64().toPercentEncoding());

    QNetworkAccessManager* manager = new QNetworkAccessManager;
    reply = manager->post(request, data.toLocal8Bit());

    connect(manager, &QNetworkAccessManager::finished, this, &Widget::onRequestFinished);
    connect(manager, &QNetworkAccessManager::finished, manager, &QNetworkAccessManager::deleteLater);
}*/

void Widget::parse()
{
    c2tProcess = new QProcess(this);

    connect(c2tProcess, SIGNAL(finished(int)), this, SLOT(onCapture2TextFinished(int)));

    QStringList args{"-i", "tmp.png"};
    c2tProcess->start("./Capture2Text/Capture2Text_CLI.exe", args);
}


void Widget::onCapture2TextFinished(int code){
    QFile::remove("tmp.png");
    if(code != 0)
        wordQueue("");

    QString word = c2tProcess->readAllStandardOutput();

    c2tProcess->deleteLater();

    wordQueue(word);
}

void Widget::onRequestFinished(){
    if(QFile::exists("tmp.png"))
        QFile::remove("tmp.png");

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());

    if(reply != nullptr)
        reply->deleteLater();

    if(!doc["error_code"].isUndefined()){
        qDebug()<<doc;
        QString message = "error_code: " + QString::number(doc["error_code"].toInt());
        message += "\nerror_msg: " + doc["error_msg"].toString();
        ui->textEdit->setText(message);
        update();
        return;
    }

    QJsonArray array = doc["words_result"].toArray();

    if(array.count()==0){
        wordQueue("");
    }else{
        QString word = array[0].toObject()["words"].toString();
        wordQueue(word);
    }
}

void Widget::wordQueue(QString word){
    word = word.remove(QRegularExpression("[^a-zA-Z0-9\\s-]+"));
    word = word.trimmed();

    if(word.isEmpty()){
        ui->textEdit->setText("(No result)");
        update();
        return;
    }

    qDebug()<<word;

    if(!hashMap.contains(word))
        word = word.toLower();

    if(hashMap.contains(word)){
        ui->textEdit->setText(hashMap[word]);
    }else{
        ui->textEdit->setText(word + " (No result)");
    }

    //Translator* t = new Translator(word);
    //connect(t, &Translator::onFinished, [=](QString html){ui->textBrowser->setHtml(html);});
    //connect(t, &Translator::onFinished, t, &Translator::deleteLater);

    setVisible(true);
    update();
}


Widget::~Widget()
{
    delete ui;
}

////////////////

//拖拽操作
void Widget::mousePressEvent(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton)
    {
        _drag = true;
        //获得鼠标的初始位置
        mouseStartPoint = event->globalPos();
        windowTopLeftPoint = this->frameGeometry().topLeft();
    }
}

void Widget::mouseMoveEvent(QMouseEvent *event)
{
    if(_drag)
    {
        QPoint distance = event->globalPos() - mouseStartPoint;
        this->move(windowTopLeftPoint + distance);
    }
}

void Widget::mouseReleaseEvent(QMouseEvent *event)
{
    if(event->button() == Qt::LeftButton)
    {
        _drag = false;
    }
}
