---
permalink: /
layout: home
---

playdict是一个极简的游戏查词工具。

![demo](./assets/imgs/demo/demo.png)

\>> [（Github Release）](https://github.com/blueloveTH/playdict/releases/latest) [（百度网盘：提取码bqz2）](https://pan.baidu.com/s/1cVgOJY4rXG1j0g8lj1GGCQ)

### 功能特性

+ 沉浸式。保持置顶，为全屏游戏进行了优化
+ 图片识别。内置OCR模型，可识别单词和短语
+ 绿色轻量。免安装，包体仅13M

### 使用说明

将`playdict-x.x.zip`解压到你的文件夹，运行`playdict.exe`

+ F1键/鼠标中键：截屏划词
+ F2键：显示或隐藏面板
+ F3键：退出程序

默认情况下，playdict会置顶窗口以保证显示在游戏界面上方。然而，部分游戏在**全屏模式**下直接操作显卡，不受窗口管理器的控制。我们提供了一种方案来解决这个问题：

1.  将游戏设置成和桌面尺寸相同的窗口模式（例如1920*1080）
2.  保持playdict在后台，点击游戏内任意区域，按下F4键
3.  这时playdict会尝试把游戏窗口转换成**全屏窗口模式**
4.  点击确认按钮，完成转换

### 性能测试

| 版本  | 样本量 | 准确率 | 平均延迟  |
| ----- | ------ | ------ | --------- |
| v0.32 | 241    | 85.71% | 300-600ms |

### TODO

-   [ ] 自定义快捷键
-   [ ] 在线更新
-   [ ] MacOS版本移植

### 开发者

[@blueloveTH](https://github.com/blueloveTH)

[@MikeBernardHan](https://github.com/MikeBernardHan)

