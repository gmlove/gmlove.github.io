---
title: 5行代码的自动评论机器人
categories:
  - 小技巧
tags:
  - 自动化
date: 2017-12-07 20:03:06
---

又到小米发布会了，这次发布会将从发布评论的人里面选人，每分钟送一台小米手机。
于是写了几行代码自动发评论，省去了手工的麻烦。娱乐一下，碰个运气。

直播地址：https://hd.mi.com/x/12041b/index.html?client_id=180100041086&masid=17409.0195

代码如下：

```javascript
// 随机选择一个当前评论列表里面的评论
var r = () => Math.floor((Math.random() * $('.livechat-list-wrapper .list li').length))
// 提取选中的评论的内容
var text = () => $($('.livechat-list-wrapper .list li')[r()]).find('.content').text()
// 使用选中的内容自动发评论
var c = () => {$('#J_chatContent').val(text());$('#J_sendChatBtn').attr('class', 'btn active');$('#J_sendChatBtn').click();}
// 生成随机的间隔时间
var rtime = () => Math.floor(Math.random() * 15000 + 5000)
// 设置一个计时器定时发评论
var st = () => stt = setTimeout(() => {c(); st()}, rtime())
st();
```

以上代码粘贴到控制台执行就可以了。

