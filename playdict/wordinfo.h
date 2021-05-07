#ifndef WORDINFO_H
#define WORDINFO_H

#include <QString>
#include <QList>
#include <QPair>

struct WordInfo{
    QString word;

    QStringList pronunciation;
    QList<QPair<QString, QString>> definition;

    explicit WordInfo(QString word){
        this->word = word;
    }

    QString noResult(){
        return word + '\n' + "(No result)";
    }

    QString simpleResult(){
        QString str = word + '\n';
        for(const auto &p : qAsConst(pronunciation))
            str += p + ' ';
        if(!pronunciation.empty())
            str += '\n';
        for(const auto &def : qAsConst(definition))
            str += def.first + '\t' + def.second + '\n';
        return str.trimmed();
    }
};

#endif // WORDINFO_H
