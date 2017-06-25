---
layout: post
title: "蓝牙知识学习"
date: 2016-03-14 16:53:40 +0800
comments: false
categories: 
- IoT
---

* 工作过程：

```
            StandBy(待机)
        Inqury(查询)      Page(寻呼)
            Authentication(配对)
            Connection(连接)
```

* 连接状态：

    - 活动状态：正在通信
    - 监听状态：随时准备通信
    - 保持状态：仅仅定时器工作，无法通信
    - 休眠模式：能耗最低，偶尔监听和检查网络信息
