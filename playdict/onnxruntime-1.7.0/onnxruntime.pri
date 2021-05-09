HEADERS += $$PWD/onnxruntime_cxx_api.h \
            $$PWD/cpu_provider_factory.h \
            $$PWD/onnxruntime_cxx_inline.h \
            $$PWD/onnxruntime_session_options_config_keys.h

LIBS += -l$$PWD/onnxruntime

INCLUDEPATH += $$PWD
