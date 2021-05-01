#include <QApplication>
#include "widget.h"
#include <QtWebView>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QSharedMemory shared("apa");
    if(shared.attach()) //共享内存被占用则直接返回
        return 0;
    shared.create(8);


    a.setQuitOnLastWindowClosed(false);

    Widget w;
    w.show();

    return a.exec();
}



