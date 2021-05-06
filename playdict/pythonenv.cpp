#include "pythonenv.h"

PythonEnv *PythonEnv::_instance = nullptr;

PythonEnv::PythonEnv() : QObject(nullptr)
{
    Py_SetPythonHome(L"python");
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.append('python/scripts')");
}

PythonEnv::~PythonEnv(){
    PyRun_SimpleString("import _clearcache");
    Py_Finalize();
}
