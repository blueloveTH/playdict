#-------------------------------------------------
#
# Project created by QtCreator 2021-04-20T21:04:23
#
#-------------------------------------------------
QT       += core gui
QT       += xml
QT       += widgets

CONFIG   += C++11
DESTDIR   = ../bin

TARGET = playdict
TEMPLATE = app

include(QHotkey-1.4.2/qhotkey.pri)
include(oescreenshot/oescreenshot.pri)
include(qhook/qhook.pri)
include(onnxruntime-1.7.0/onnxruntime.pri)

SOURCES += main.cpp\
    bingdict.cpp \
    detector.cpp \
    recognizer.cpp \
    widget.cpp

HEADERS  += \
    bingdict.h \
    detector.h \
    onnxsession.h \
    recognizer.h \
    uidefbar.h \
    widget.h \
    httplib.h \
    wordinfo.h

FORMS += \
    widget.ui
