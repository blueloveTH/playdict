#ifndef SCREENUTIL_H
#define SCREENUTIL_H


#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <QString>

class ScreenUtil
{
    static float scalingFactor_;
    int static getDesktopHeight();
public:
    ScreenUtil();

    float static scalingFactor();
    void static setAutoScalingFactor(float a=1.0);
};

#endif // SCREENUTIL_H
