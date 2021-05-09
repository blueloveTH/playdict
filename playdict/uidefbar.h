#ifndef UIDEFBAR_H
#define UIDEFBAR_H

#include <QWidget>
#include <QTextBrowser>
#include <QLabel>
#include <QScrollBar>
#include <QScrollArea>

class UiDefinitionBar{

    QLabel* defFirst;
    QTextBrowser* defSecond;
public:
    UiDefinitionBar(QWidget* parent, int x, int y, const QPair<QString, QString>& content){
        defFirst = new QLabel(parent);
        defFirst->setGeometry(QRect(x, y, 71, 31));
        defFirst->setContextMenuPolicy(Qt::NoContextMenu);
        if(content.first == QString("网络"))
            defFirst->setStyleSheet(QString::fromUtf8("font: bold; background: rgb(75,75,75);"));
        else
            defFirst->setStyleSheet(QString::fromUtf8("font: bold; background: rgb(14,99,156);"));
        defFirst->show();

        defSecond = new QTextBrowser(parent);
        defSecond->setEnabled(true);
        defSecond->setGeometry(QRect(x+65+3, y, 255-3, 40));
        defSecond->setStyleSheet(QString::fromUtf8("border: none;"));
        defSecond->setUndoRedoEnabled(false);
        defSecond->setReadOnly(true);
        defSecond->setTextInteractionFlags(Qt::NoTextInteraction);
        defSecond->setOpenLinks(false);
        defSecond->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        defSecond->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        defSecond->show();

        /****************************/

        QString c1 = content.first;
        c1 = " " + c1.replace("modalv.", "m.v.") + " ";
        defFirst->setText(c1);
        defFirst->adjustSize();

        defSecond->setText(content.second);
        defSecond->document()->setDocumentMargin(0);
        defSecond->document()->setTextWidth(defSecond->width());
        defSecond->setFixedHeight(defSecond->document()->size().height());
    }

    ~UiDefinitionBar(){
        delete defFirst;
        delete defSecond;
    }

    int height(){
        int h_1 = defFirst->rect().height();
        int h_2 = defSecond->rect().height();
        return fmax(h_1, h_2);
    }
};

#endif // UIDEFBAR_H
