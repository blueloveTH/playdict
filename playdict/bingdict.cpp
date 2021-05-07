#include "bingdict.h"

BingDict::BingDict(){
    connect(this, &BingDict::finished, [&]{_isReady=true;});

    client = new httplib::Client("http://cn.bing.com");
    client->set_connection_timeout(3);

    //client->set_keep_alive(true);
    //client->set_decompress(true);
    //client->set_default_headers({
    //  { "Accept-Encoding", "gzip, deflate, br" }
    //});

    qRegisterMetaType<WordInfo>("WordInfo");
}

QList<QStringList> BingDict::findAll(const QString& pattern_str, const QString& text, int offset=0){
    QList<QStringList> results;
    QRegExp pattern(pattern_str);
    pattern.setMinimal(true);
    while(true){
        int pos = pattern.indexIn(text, offset);
        if(pos < 0) break;
        QStringList caps = pattern.capturedTexts();
        if(caps.count()>1) caps.removeFirst();
        offset = pos + pattern.matchedLength();
        results.append(caps);
    }

    return results;
}

QList<int> BingDict::findAllIndex(const QString& pattern_str, const QString& text, int offset=0){
    QList<int> results;
    QRegExp pattern(pattern_str);
    pattern.setMinimal(true);
    while(true){
        int pos = pattern.indexIn(text, offset);
        if(pos < 0) break;
        offset = pos + pattern.matchedLength();
        results.append(pos);
    }

    return results;
}

QString BingDict::subStringDiv(QString text, int startPos){
    QString pattern_0 = "<div";
    QString pattern_1 = "</div>";
    QList<int> results_0 = findAllIndex(pattern_0, text, startPos);
    QList<int> results_1 = findAllIndex(pattern_1, text, startPos);

    if(results_0.empty() || results_1.empty())
        return "";
    if(results_0[0] > results_1[0])
        return "";

    int i_0 = 1;
    int i_1 = 0;

    int indent = 1;

    while(true){
        if(i_0<results_0.count() && (results_0[i_0]<results_1[i_1])){
            indent++;
            i_0++;
        }else{
            indent--;
            if(indent==0)
                return text.mid(results_0[0], results_1[i_1] + pattern_1.count() - results_0[0]);
            i_1++;
            if(i_1>=results_1.count())
                break;
        }
    }

    return "";
}

void BingDict::onReply(QByteArray bytes){
    QString html = QString(bytes);

    WordInfo wordinfo(current_query);

    /// clean
    QString cls_pattern = "<div class=\"client_def_container\">";
    int idx = html.indexOf(cls_pattern);
    if (idx < 0){
        emit finished(wordinfo);
        return;
    }

    /// pronunciation
    auto pron_matches = findAll("<div class=\"client_def_hd_pn\" lang=\"en\">.*</div>", html.left(idx));
    QDomDocument doc;
    for(int i=0;i<pron_matches.count();i++){
        doc.setContent(pron_matches[i][0]);
        QDomNodeList list=doc.elementsByTagName("div");
        QString text = list.at(0).toElement().text();
        if(text.contains('['))
            wordinfo.pronunciation.append(text);
    }

    /// def container
    html = subStringDiv(html, idx);

    /// definition
    QList<int> defMatches = findAllIndex("<div class=\"client_def_bar\">", html);

    QRegExp titlePattern("<span class=\"client_def_title[_web]*\">(.*)</span>");
    titlePattern.setMinimal(true);
    for(auto idx : defMatches){
        QString tmp_html = subStringDiv(html, idx);
        int pos = titlePattern.indexIn(tmp_html);
        if(pos < 0) continue;
        QString title = titlePattern.cap(1);

        pos += titlePattern.matchedLength();
        tmp_html = tmp_html.replace("<a class=", "<span class=");
        tmp_html = tmp_html.replace("</a>", "</span>");
        auto listMatches = findAll("<span class=\"client_def_list.*\">(.*)</span>", tmp_html, pos);

        QString content = "";
        for(auto str_list : listMatches)
            content += str_list[0];

        wordinfo.definition.append(QPair<QString, QString>(title, content));
    }

    emit finished(wordinfo);
}

void BingDict::query(QString q){
    _isReady = false;
    current_query = q = q.trimmed();

    QtConcurrent::run([=]{
        QString url = "/dict/clientsearch?mkt=zh-CN&setLang=zh&q=";
        url += q;
        std::string body;
        bool isOk = false;
        client->Get(url.toLatin1().data(),
                               [&](const httplib::Response &res) {
                                 isOk = res.status == 200;
                                 return isOk;
                               },
                               [&](const char *data, size_t data_length) {
                                 body.append(data, data_length);
                                 return body.size() < 85000;
                               });
        const char *result;
        if(isOk)
            result = body.data();
        else
            result = QString("(Connection error)").toLatin1().data();
        onReply(QByteArray(result));
    });
}

