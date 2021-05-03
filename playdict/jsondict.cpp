#ifndef JSONDICT_CPP
#define JSONDICT_CPP

#include <QString>
#include <QFile>
#include <QHash>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QtDebug>

class JsonDict{
    QHash<QString, QString> hashMap;
    QJsonArray dicArray;
    bool _loaded = false;

public:
    void load(const QString& filename){
        QFile file(filename);
        file.open(QIODevice::ReadOnly);

        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        file.close();

        dicArray = doc["data"].toArray();
        for(auto i=dicArray.constBegin(); i!=dicArray.constEnd(); i++){
            auto a = i->toArray();
            hashMap[a[0].toString()] = a[1].toString();
        }

        qDebug() << "Json dict loaded." << hashMap.count();
        _loaded = true;
    }

    bool isLoaded(){
        return _loaded;
    }

    QString query(QString word){
        if(!isLoaded())
            return "(Dictionary hasn't been loaded)";
        word = word.remove(QRegularExpression("[^a-zA-Z0-9\\s-]+"));
        word = word.trimmed();

        if(word.isEmpty()) return "(No result)";

        if(!hashMap.contains(word)) word = word.toLower();

        if(hashMap.contains(word)){
            return hashMap[word];
        }else{
            return word + " (No result)";
        }
    }
};

#endif
