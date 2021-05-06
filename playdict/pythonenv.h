#ifndef PYTHONENV_H
#define PYTHONENV_H

#include <QObject>
#include <QtDebug>

#pragma push_macro("slots")
#undef slots
#include "Python.h"
#pragma pop_macro("slots")

class PythonEnv : public QObject
{
    Q_OBJECT

    static PythonEnv *_instance;

    explicit PythonEnv();
public:
    static PythonEnv* Instance(){return _instance;}
    static void Initialize(){
        _instance = new PythonEnv();
    }

    static void Finalize(){
        delete _instance;
    }

    ~PythonEnv();



signals:

};

#endif // PYTHONENV_H
