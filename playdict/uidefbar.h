#ifndef UIDEFBAR_H
#define UIDEFBAR_H

#include <QWidget>
#include <QTextBrowser>
#include <QLabel>

class UiDefinitionBar{

    QLabel* defFirst;
    QTextBrowser* defSecond;
public:
    UiDefinitionBar(QWidget* parent, int x, int y, const QPair<QString, QString>& content){
        defFirst = new QLabel(parent);
        defFirst->setGeometry(QRect(x, y, 71, 30));
        QFont font;
        font.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        font.setPointSize(10);
        font.setBold(true);
        font.setItalic(false);
        font.setWeight(75);
        defFirst->setFont(font);
        defFirst->setContextMenuPolicy(Qt::NoContextMenu);
        if(content.first == QString("网络"))
            defFirst->setStyleSheet(QString::fromUtf8("background: rgb(75,75,75);"));
        else
            defFirst->setStyleSheet(QString::fromUtf8("background: rgb(14,99,156);"));
        defFirst->show();

        defSecond = new QTextBrowser(parent);
        defSecond->setEnabled(true);
        defSecond->setGeometry(QRect(x+80, y, 299, 40));
        QFont font1;
        font1.setFamily(QString::fromUtf8("\345\276\256\350\275\257\351\233\205\351\273\221"));
        font1.setPointSize(10);
        defSecond->setFont(font1);
        defSecond->setStyleSheet(QString::fromUtf8("border: none;"));
        defSecond->setUndoRedoEnabled(false);
        defSecond->setReadOnly(true);
        defSecond->setTabStopWidth(60);
        defSecond->setTextInteractionFlags(Qt::NoTextInteraction);
        defSecond->setOpenLinks(false);
        defSecond->show();

        /****************************/

        defFirst->setText(" " + content.first);
        defSecond->setText(content.second);
        defFirst->setFixedWidth(QTextDocument(" " + content.first).size().width());

        defSecond->setFixedHeight(defSecond->document()->size().height());
        defSecond->document()->setDocumentMargin(2);
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
