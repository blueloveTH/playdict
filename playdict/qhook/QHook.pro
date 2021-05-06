#-------------------------------------------------
#
# Project created by QtCreator 2015-01-04T08:22:48
#
#-------------------------------------------------

QT       -= gui

TARGET = QHook
TEMPLATE = lib
CONFIG += staticlib

win32:LIBS += -luser32

SOURCES += qhook.cpp \
    qhookmouseevent.cpp \
    mousehook.cpp


HEADERS += qhook.h \
    qhookmouseevent.h \
    mousehook.h