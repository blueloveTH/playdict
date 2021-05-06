#ifndef QHOOK_H
#define QHOOK_H

#include <QObject>


#include "qhookmouseevent.h"
#include "mousehook.h"
#include <QDebug>


class QHook : public QObject
{
    Q_OBJECT
public:
    explicit QHook(bool autoHook);
    ~QHook();

    static QHook* Instance();

    bool isMouseHooked();

    virtual bool mouseMoveEvent(QHookMouseEvent *event);
    virtual bool mousePressEvent(QHookMouseEvent *event);
    virtual bool mouseReleaseEvent(QHookMouseEvent *event);

signals:
    void mousePressed(QHookMouseEvent*);

private:
    MouseHook *p_MouseHook;


public slots:
    void hookMouse();
    void unhookMouse();
};

#endif // QHOOK_H
