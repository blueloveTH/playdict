#include <QApplication>
#include "widget.h"
#include <QtWebView>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    /// Force single instance
    QSharedMemory shared("apa");
    if(shared.attach())
        return 0;
    shared.create(8);

    a.setQuitOnLastWindowClosed(false);

    Widget w;
    w.show();

    return a.exec();
}



