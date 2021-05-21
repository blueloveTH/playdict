#-------------------------------------------------
#
# Project created by QtCreator 2021-04-20T21:04:23
#
#-------------------------------------------------
QT       += core gui
QT       += xml
QT       += widgets

CONFIG   += C++17
DESTDIR   = ../bin

TARGET = playdict
TEMPLATE = app

include(QHotkey-1.4.2/qhotkey.pri)
include(qhook/qhook.pri)
include(oescreenshot/oescreenshot.pri)
include(onnxruntime-1.7.0/onnxruntime.pri)

include(bingdict/bingdict.pri)
include(pipeline/pipeline.pri)
include(edlib/edlib.pri)

RC_ICONS = favicon.ico


SOURCES += main.cpp\
    widget.cpp

HEADERS  += \
    uidefbar.h \
    widget.h \

FORMS += \
    widget.ui

RESOURCES += \
    main.qrc
