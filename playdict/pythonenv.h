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

    explicit PythonEnv(QObject *parent = nullptr);
public:
    static PythonEnv* Instance(){
        if(_instance == nullptr)
            _instance = new PythonEnv;
        return _instance;
    }

    ~PythonEnv(){Py_Finalize();}



signals:

};

#endif // PYTHONENV_H
