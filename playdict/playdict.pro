#-------------------------------------------------
#
# Project created by QtCreator 2021-04-20T21:04:23
#
#-------------------------------------------------
QT       += core gui
QT       += xml
QT       += widgets

CONFIG   += C++14
DESTDIR   = ../bin

TARGET = playdict
TEMPLATE = app

include(QHotkey-1.4.2/qhotkey.pri)
include(oescreenshot/oescreenshot.pri)
include(qhook/qhook.pri)

SOURCES += main.cpp\
    bingdict.cpp \
    pythonenv.cpp \
    recognizer.cpp \
    widget.cpp

HEADERS  += \
    bingdict.h \
    pythonenv.h \
    recognizer.h \
    widget.h

FORMS += \
    widget.ui

INCLUDEPATH += D:\miniconda\envs\playdict_qt\include

LIBS += -LD:\miniconda\envs\playdict_qt\libs -lpython38
