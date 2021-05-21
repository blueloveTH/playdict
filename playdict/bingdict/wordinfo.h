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

    bool hasResult() const{
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

    static WordInfo helpWord(){
        WordInfo wi("playdict");
        wi.pronunciation = QStringList();
        wi.pronunciation.append("(Update: May 21, 2021)");
        //wi.pronunciation.append(QString(u8"\u7f8e") + ": ['pleɪˌdɪkt]");
        //wi.pronunciation.append(QString(u8"\u82f1") + ": ['pleɪˌdɪkt]");
        wi.definition.append(QPair<QString, QString>(QString("MM"), QString("Screenshot (Mid Mouse)")));
        wi.definition.append(QPair<QString, QString>(QString("F1"), QString("Screenshot")));
        wi.definition.append(QPair<QString, QString>(QString("F2"), QString("Hide or Show")));
        wi.definition.append(QPair<QString, QString>(QString("F3"), QString("Exit")));
        return wi;
    }
};

#endif // WORDINFO_H
