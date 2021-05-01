#ifndef COMMONHELPER_H
#define COMMONHELPER_H
#include <QString>
#include <QRect>

class QWidget;

/**
 * @class : OECommonHelper
 * @brief : 通用助手
 * @note  : 完成一些比较常用又通用的功能
 */
class OECommonHelper
{
public:

    /**
    * @brief : 设置QSS文件
    * @param : style 文件名
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    * @remark: 如果是qrc路径，请带上 qrc:/
    **/
    static void setStyle(const QString &style);

    /**
    * @brief : 设置语言包
    * @param : language 语言包的文件名
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    **/
    static void setLanguagePack(const QString& language);

    /**
    * @brief : 将窗口移动到中心
    * @param : widget 要移动的窗口
    * @param : parentRect 矩形几何数据
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    **/
    static void moveCenter(QWidget* widget, QRect parentRect = {});


    /**
    * @brief : 获得当前界面与开发时的界面之间的横向倍率
    * @return: float 倍率
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    **/
    static const float& getWindowWidthMultiplyingPower(void);


    /**
    * @brief : 获得当前界面与开发时的界面之间的纵向倍率
    * @return: float 倍率
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    **/
    static const float& getWindowHeightMultiplyingPower(void);

protected:

    /**
    * @brief : 更新窗口倍率
    * @author: 陈鲁勇
    * @date  : 2017年04月10日
    **/
    static void upWindowSizeMultiplyingPower(void);


private:
    /// 窗口横向倍率
    static float widthMultiplyingPower_;
    /// 窗口纵向倍率
    static float heightMultiplyingPower_;
};

#endif /// COMMONHELPER_H
