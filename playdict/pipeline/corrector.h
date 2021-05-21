#ifndef CORRECTOR_H
#define CORRECTOR_H

#include <QString>
#include <QFile>
#include <QDebug>
#include "edlib.h"

class Corrector
{
    QList<std::string> allWords;
public:
    Corrector(){
        QFile file(":/others/res/corrector_dict.txt");
        file.open(QIODevice::ReadOnly);
        QStringList _1 = QString(file.readAll()).split(',');
        for(const QString &w : qAsConst(_1)) allWords.append(w.toStdString());
        file.close();
    }

    int editDistance(const std::string& query, const std::string& target){
        EdlibAlignResult result = edlibAlign(query.data(), query.size(), target.data(), target.size(), edlibDefaultAlignConfig());
        if (result.status == EDLIB_STATUS_OK)
            return result.editDistance;
        edlibFreeAlignResult(result);
        return 999;
    }

    QString correct(QString src){
        std::string probStr = "";
        std::string std_src = src.toLower().toStdString();
        for(auto i=allWords.constBegin(); i!=allWords.constEnd(); i++){
            float dis = editDistance(*i, std_src);
            if(dis == 0)
                return src;
            if(dis == 1)
                probStr = *i;
        }

        if(probStr.empty()) return src;
        return QString::fromStdString(probStr);
    }

};

#endif // CORRECTOR_H
