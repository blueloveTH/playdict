#ifndef OESCREENSHOT_H
#define OESCREENSHOT_H

#include <memory>
#include <QRect>
#include <QWidget>

class OEScreen;

class OEScreenshot : public QWidget {
    Q_OBJECT

signals:
    void onScreenshot();
public:
    explicit OEScreenshot(QWidget *parent = 0);
    ~OEScreenshot(void);

    static OEScreenshot *Instance(void);

    static void destroy(void);

protected:
    virtual void mousePressEvent(QMouseEvent *);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void paintEvent(QPaintEvent *);

private:
    void initGlobalScreen(void);
    void createScreen(const QPoint &pos);
    void destroyScreen(void);

    /**
     * @brief : 获得当前屏幕的大小
     * @note  : 这个函数是支持多屏幕的，示例：双屏幕 QRect（-1920, 0, 3840, 1080）
     * @return: 返回 QRect 引用
     * @date  : 2017年04月15日
     */
    const QRect& getScreenRect(void);


private:
    /// 截屏窗口是否已经展示
    bool                        isLeftPressed_;
    /// 当前桌面屏幕的矩形数据
    QRect desktopRect_;
    /// 屏幕暗色背景图
    std::shared_ptr<QPixmap>    backgroundScreen_;
    /// 屏幕原画
    std::shared_ptr<QPixmap>    originPainting_;
    /// 截图屏幕
    std::shared_ptr<OEScreen>   screenTool_;
    /// 截屏实例对象
    static OEScreenshot         *self_;
};


/**
 * @class : OEScreen
 * @brief : 截图器
 * @note  : 主要关乎图片的编辑与保存
*/
class OEScreen : public QWidget {
    Q_OBJECT
protected:

    /// 内边距，决定拖拽的触发。
    const int PADDING_ = 6;

public:

    explicit OEScreen(std::shared_ptr<QPixmap> originPainting, QPoint pos, QWidget *parent = 0);

    ~OEScreen() { }

protected:
    virtual void paintEvent(QPaintEvent *);

public slots:
    void onMouseChange(int x,int y);
    void onSaveScreen(void);

private:
    /// 起点
    QPoint          originPoint_;
    /// 屏幕原画
    std::shared_ptr<QPixmap> originPainting_;
    /// 当前窗口几何数据 用于绘制截图区域
    QRect           currentRect_;
};



#endif /// OESCREENSHOT_H
