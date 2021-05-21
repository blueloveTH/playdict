#ifndef SCREENUTIL_H
#define SCREENUTIL_H


#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <QString>
#include <QWidget>

class ScreenUtil
{
    static float scalingFactor_;
    int static getDesktopHeight();
public:
    ScreenUtil();

    float static scalingFactor();
    void static setAutoScalingFactor(float a=1.0);
    void static setWindowFlags(QWidget* widget);
};

#endif // SCREENUTIL_H
