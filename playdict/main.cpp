#include <QApplication>
#include <QSharedMemory>
#include "widget.h"

int main(int argc, char *argv[])
{
    /// Force single instance
    QSharedMemory shared("apa");
    if(shared.attach())
        return 0;
    shared.create(8);

    ScreenUtil::setAutoScalingFactor();
    QApplication a(argc, argv);
    a.setQuitOnLastWindowClosed(false);

    Widget w(&a);
    w.show();
    return a.exec();
}



