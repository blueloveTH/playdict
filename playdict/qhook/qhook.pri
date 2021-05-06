CONFIG += C++11

win32:LIBS += -luser32

SOURCES += $$PWD/qhook.cpp \
    $$PWD/qhookmouseevent.cpp \
    $$PWD/mousehook.cpp

HEADERS += $$PWD/qhook.h \
    $$PWD/qhookmouseevent.h \
    $$PWD/mousehook.h

INCLUDEPATH += $$PWD
