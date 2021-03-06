#include "screenutil.h"

float ScreenUtil::scalingFactor_ = 1.0;

ScreenUtil::ScreenUtil()
{

}


QSize ScreenUtil::getDesktopWH(){
    HWND hWnd = GetDesktopWindow();
    HMONITOR hMonitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);

    MONITORINFOEX miex;
    miex.cbSize = sizeof(miex);
    GetMonitorInfo(hMonitor, &miex);
    //int cxLogical = (miex.rcMonitor.right - miex.rcMonitor.left);
    //int cyLogical = (miex.rcMonitor.bottom - miex.rcMonitor.top);

    DEVMODE dm;
    dm.dmSize = sizeof(dm);
    dm.dmDriverExtra = 0;
    EnumDisplaySettings(miex.szDevice, ENUM_CURRENT_SETTINGS, &dm);
    int cxPhysical = dm.dmPelsWidth;
    int cyPhysical = dm.dmPelsHeight;
    return QSize(cxPhysical, cyPhysical);
}


float ScreenUtil::scalingFactor(){
    return scalingFactor_;
}

void ScreenUtil::setAutoScalingFactor(float a){
    scalingFactor_ = getDesktopWH().height() / 1080.0 * a;
    scalingFactor_ = fmax(1.0, scalingFactor_);
    qputenv("QT_SCALE_FACTOR_ROUNDING_POLICY", "1");
    qputenv("QT_SCALE_FACTOR", QString::number(scalingFactor_).toLatin1());
}

void ScreenUtil::setWindowFlags(QWidget *widget){
    /// Set windows
    Qt::WindowFlags flags = widget->windowFlags();
    flags |= Qt::WindowStaysOnTopHint;
    flags |= Qt::FramelessWindowHint;
    flags |= Qt::BypassWindowManagerHint;
    widget->setWindowFlags(flags);

    /// No focus
    HWND wid = (HWND)(widget->winId());
    SetWindowLong(wid, GWL_EXSTYLE, GetWindowLong(wid, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_COMPOSITED);
}
