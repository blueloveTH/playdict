#include "translator.h"

Translator::Translator(const QString& queue)
{
    QNetworkRequest request;
    request.setUrl(QUrl("https://cn.bing.com/dict/clientsearch?mkt=zh-CN&setLang=zh&q=" + queue));

    QNetworkAccessManager* manager = new QNetworkAccessManager;

    QObject::connect(manager, &QNetworkAccessManager::finished, manager, &QNetworkAccessManager::deleteLater);

    QObject::connect(manager, &QNetworkAccessManager::finished, this, &Translator::onRequestFinished);
    reply = manager->get(request);
}

void Translator::onRequestFinished(){
    QString html = QString(reply->readAll());
    emit onFinished(html);
}
