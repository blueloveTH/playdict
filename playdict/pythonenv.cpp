#include "pythonenv.h"

PythonEnv *PythonEnv::_instance = nullptr;

PythonEnv::PythonEnv(QObject *parent) : QObject(parent)
{
    Py_SetPythonHome(L"python");
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.append('python/scripts')");
    PyRun_SimpleString("import bingdict");
}
