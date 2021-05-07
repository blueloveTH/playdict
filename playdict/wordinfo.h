#ifndef WORDINFO_H
#define WORDINFO_H

#include <QString>
#include <QList>
#include <QPair>

struct WordInfo{

    QString word;

    QStringList pronunciation;
    QList<QPair<QString, QString>> definition;

    WordInfo(){}

    WordInfo(QString word){
        this->word = word;
    }

    bool hasResult(){
        return !definition.empty();
    }

    QString noResult(){
        return word + '\n' + "(No result)";
    }

    QString pronResult() const{
        QString str = "";
        for(const auto &p : qAsConst(pronunciation))
            str += p + "  ";
        return str;
    }

    QString simpleResult(){
        QString str = word + '\n';
        if(!pronunciation.empty())
            str += pronResult() + '\n';
        for(const auto &def : qAsConst(definition))
            str += def.first + '\t' + def.second + '\n';
        return str.trimmed();
    }
};

#endif // WORDINFO_H
