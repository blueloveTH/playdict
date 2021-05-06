#-------------------------------------------------
#
# Project created by QtCreator 2021-04-20T21:04:23
#
#-------------------------------------------------
QT       += core gui
QT       += network
QT       += xml

CONFIG   += C++17
DESTDIR   = ../bin
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = playdict
TEMPLATE = app

include(QHotkey-1.4.2/qhotkey.pri)
include(oescreenshot/oescreenshot.pri)
include(qhook/qhook.pri)

SOURCES += main.cpp\
    bingdict.cpp \
    recognizer.cpp \
    widget.cpp

HEADERS  += \
    bingdict.h \
    recognizer.h \
    widget.h

FORMS += \
    widget.ui

