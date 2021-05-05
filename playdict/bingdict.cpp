#include "bingdict.h"

QList<QStringList> BingDict::findAll(QRegExp& pattern, const QString& text){
    QList<QStringList> results;
    pattern.setMinimal(true);
    int offset = 0;
    while(true){
        int pos = pattern.indexIn(text, offset);
        if(pos < 0) break;
        QStringList caps = pattern.capturedTexts();
        caps.removeFirst();
        offset = pos + pattern.matchedLength();
        results.append(caps);
    }

    return results;
}


void BingDict::onRequestFinished(){
    QString html = QString(reply->readAll());
    QString return_str = current_query + '\n';

    /// clean
    QString cls_pattern = "<div class=\"client_def_container\">";
    int idx = html.indexOf(cls_pattern);
    if (idx < 0){
        emit finished(return_str + "(No result)");
        return;
    }

    idx = html.indexOf(cls_pattern, idx+cls_pattern.count());
    if (idx > 0)
        html = html.left(idx);

    /// pronunciation
    auto pron_pattern = QRegExp("<div class=\"client_def_hd_pn\" lang=\"en\">(.*)</div>");

    auto pron_matches = findAll(pron_pattern, html);
    for(int i=0;i<pron_matches.count();i++){
        return_str += pron_matches[i][0].replace("&#160;", " ") + ' ';
    }

    if(!return_str.isEmpty())
        return_str += '\n';

    /// define
    auto def_pattern = QRegExp("<div class=\"client_def_bar\">.*<span class=\"client_def_title\">(.*)</span>.*<span class=\"client_def_list_word_bar\">(.*)</span>");
    auto def_matches = findAll(def_pattern, html);

    for(int i=0;i<def_matches.count();i++){
        return_str += def_matches[i][0] + ' ' + def_matches[i][1] + '\n';
    }

    emit finished(return_str);
}

void BingDict::query(QString q){
    _isReady = false;
    current_query = q = q.trimmed();

    QNetworkRequest request;
    request.setUrl(QUrl("https://cn.bing.com/dict/clientsearch?mkt=zh-CN&setLang=zh&q=" + q));

    QNetworkAccessManager* manager = new QNetworkAccessManager;

    connect(manager, &QNetworkAccessManager::finished, manager, &QNetworkAccessManager::deleteLater);
    connect(manager, &QNetworkAccessManager::finished, this, &BingDict::onRequestFinished);

    reply = manager->get(request);
}

