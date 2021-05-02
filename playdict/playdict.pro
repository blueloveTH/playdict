#-------------------------------------------------
#
# Project created by QtCreator 2021-04-20T21:04:23
#
#-------------------------------------------------
QT       += core gui
QT       += network
QT       += webview

CONFIG   += C++11
DESTDIR   = ../bin
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = playdict
TEMPLATE = app
INCLUDEPATH += "oescreenshot/"
include(qxt/qxt.pri)

SOURCES += main.cpp\
    jsondict.cpp \
    oescreenshot/oecommonhelper.cpp \
    oescreenshot/oescreenshot.cpp \
    recognizer.cpp \
    widget.cpp

HEADERS  += \
    oescreenshot/oecommonhelper.h \
    oescreenshot/oescreenshot.h \
    recognizer.h \
    widget.h

FORMS += \
    widget.ui

