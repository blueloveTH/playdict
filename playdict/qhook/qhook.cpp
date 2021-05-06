#include "qhook.h"

QHook *hook = nullptr;

QHook* QHook::Instance(){
    if(hook == nullptr)
        hook = new QHook(true);
    return hook;
}

QHook::QHook(bool autoHook)
{
    p_MouseHook = new MouseHook();
    if(autoHook)
        hookMouse();
}

QHook::~QHook()
{
    p_MouseHook->unhook();
}

void QHook::hookMouse()
{
    p_MouseHook->hook();
}

void QHook::unhookMouse()
{
    p_MouseHook->unhook();
}

bool QHook::isMouseHooked(){ return p_MouseHook->isHooked(); }

/*!
 * The functions below may be overridden to determin the event
 * and wheather or not it should go through or not
 *
 * Returning `true` on an event will allow the event to go through
 * while returning `false` will block the event
 */

bool QHook::mouseMoveEvent(QHookMouseEvent *event){ return true; }
bool QHook::mousePressEvent(QHookMouseEvent *event){
    emit mousePressed(event);
    return true;
}
bool QHook::mouseReleaseEvent(QHookMouseEvent *event){ return true; }
