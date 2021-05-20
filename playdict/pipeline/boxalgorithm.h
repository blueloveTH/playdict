#ifndef BOXALGORITHM_H
#define BOXALGORITHM_H

#include <QPoint>
#include <vector>
#include <queue>
#include <QObject>
#include <QtDebug>
#include <QImage>
#include <algorithm>

struct MatArray{
    const int* mat;
    int height, width;
    MatArray(const int* mat, int height, int width){
        this->mat = mat;
        this->height = height;
        this->width = width;
    }

    int get(int i, int j) const{
        return mat[i*width+j];
    }
};

/// See reference: https://www.jianshu.com/p/1ace30997163
class ConnectionDetector
{
public:
    int m;
    int n;

    bool isvalid(int i, int j, const MatArray& matrix, std::vector<std::vector<bool>>& mask) {
        return i >= 0 && i < m && j >= 0 && j < n && !mask[i][j] && matrix.get(i,j) == 1;
    }

    void add(int i, int j, const MatArray& matrix, std::queue<QPoint>& q, std::vector<std::vector<bool>>& mask) {
        if (isvalid(i, j, matrix, mask)) {
            q.push(QPoint(i, j));
            mask[i][j] = true;
        }
    }

    std::vector<std::vector<QPoint>> bwlabel(const MatArray& matrix) {
        m = matrix.height;
        n = matrix.width;
        std::vector<std::vector<QPoint>> res;
        std::vector<QPoint> tmp;
        std::vector<std::vector<bool>> mask(m, std::vector<bool>(n, false));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (mask[i][j] || matrix.get(i, j) == 0)
                    continue;
                tmp.clear();
                std::queue<QPoint> q;
                q.push(QPoint(i, j));
                mask[i][j] = true;
                while (!q.empty()) {
                    QPoint t = q.front();
                    q.pop();
                    tmp.push_back(t);
                    add(t.x() - 1, t.y(), matrix, q, mask);
                    add(t.x() + 1, t.y(), matrix, q, mask);
                    add(t.x(), t.y() - 1, matrix, q, mask);
                    add(t.x(), t.y() + 1, matrix, q, mask);
                }
                res.push_back(tmp);
            }
        }
        return res;
    }
};


struct Rect{
    float xMin, yMin, xMax, yMax;
    Rect(float xMin, float yMin, float xMax, float yMax){
        this->xMin = xMin;
        this->yMin = yMin;
        this->xMax = xMax;
        this->yMax = yMax;
    }

    Rect(){
        xMin = xMax = yMin = yMax = -1;
    }

    bool isEmpty() const{
        return xMin == -1;
    }

    bool contains(float x, float y) const{
        bool _1 = x >= xMin && x <= xMax;
        bool _2 = y >= yMin && y <= yMax;
        return _1 && _2;
    }

    float distance(float x, float y) const{
        QPointF diff = center() - QPointF(x, y);
        return sqrt(QPointF::dotProduct(diff, diff));
    }

    QPointF center() const{
        float _1 = (xMin+xMax) * 0.5;
        float _2 = (yMin+yMax) * 0.5;
        return QPointF(_1, _2);
    }

    float ySpan() const{
        return yMax - yMin + 1;
    }

    float xSpan() const{
        return xMax - xMin + 1;
    }

    float area() const{
        return xSpan() * ySpan();
    }

    Rect united(const Rect& other){
        Rect rect;
        rect.xMin = fmin(xMin, other.xMin);
        rect.xMax = fmax(xMax, other.xMax);
        rect.yMin = fmin(yMin, other.yMin);
        rect.yMax = fmax(yMax, other.yMax);
        return rect;
    }

    QString repr() const{
        return QString("Rect(%1, %2, %3, %4)").arg(xMin).arg(yMin).arg(xMax).arg(yMax);
    }
};


inline QList<Rect> getAnchorBox01(const MatArray& mat){
    QList<Rect> results;

    ConnectionDetector cd;
    auto res = cd.bwlabel(mat);

    for(uint i=0; i<res.size(); i++){
        int minX=1e6, minY=1e6, maxX=-1e6, maxY=-1e6;
        for(uint j=0; j<res[i].size(); j++){
            int x_ = res[i][j].x();
            int y_ = res[i][j].y();
            minX = fmin(x_, minX);
            minY = fmin(y_, minY);
            maxX = fmax(x_, maxX);
            maxY = fmax(y_, maxY);
        }
        auto rect = Rect(minX, minY, maxX, maxY);
        if(rect.area() > 16*16)
            results.append(rect);
    }
    return results;
}


struct RegionInfo{
    int index;
    float distance;

    RegionInfo(int index, float distance){
        this->index = index;
        this->distance = distance;
    }
};

inline Rect selectMainBox(QList<Rect> results, int x, int y){
    QList<RegionInfo> infoList;
    for(int i=0; i<results.size(); i++){
        float dis = results.at(i).distance(x, y);
        infoList.append(RegionInfo(i, dis));
    }

    if(infoList.empty()) return Rect();

    std::sort(infoList.begin(), infoList.end(),
              [=](const RegionInfo& lhs, const RegionInfo rhs){return lhs.distance<rhs.distance;});

    Rect rect = results[infoList[0].index];

    /*
    for(auto i: infoList){
        qDebug() << results[i.index].repr();
        qDebug() << i.index << i.distance;
    }*/

    if(results.size() > 1){
        Rect secondRect = results[infoList[1].index];
        bool _1 = rect.center().x() - secondRect.center().x() < rect.xSpan() * 0.25;
        bool _2 = (infoList[0].distance - infoList[1].distance) <= infoList[0].distance * 0.25;
        if(_1 && _2){
            //qDebug() << "united.";
            rect = rect.united(secondRect);
        }
    }

    return rect;
}

inline Rect getMainBox(const MatArray& mat){
    QList<Rect> results = getAnchorBox01(mat);
    return selectMainBox(results, mat.height*0.5, mat.width*0.5);
}

#endif // BOXALGORITHM_H
