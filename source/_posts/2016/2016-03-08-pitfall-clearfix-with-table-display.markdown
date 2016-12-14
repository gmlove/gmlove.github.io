---
layout: post
title: "在clearfix中使用`display:table`"
date: 2016-03-08 17:22:40 +0800
comments: true
categories:
- frontend
tags:
- css
---

当子元素是浮动布局时，父元素无法获取到正确的宽高，这种情况常常使用clearfix方案来解决。

** 示例如下：**

```html
<div> <!-- 父元素无法获取到正确的尺寸 -->
    <div style="float:left; width:100px; height:100px;"></div>
</div>
```
<!--more-->


** clearfix方案：（参考[bootstrap文档](https://getbootstrap.com/css/#helper-classes-clearfix)） **

```html
.clearfix:before, .clearfix:after { display: table; content: " "; }
.clearfix:after { clear: both; }
<div class="clearfix"> <!-- 父元素可以获取到正确的尺寸 -->
    <div style="float:left; width:100px; height:100px;"></div>
</div>
```


但是`display: table`事实上会产生一个额外的问题，因为table的布局会使父元素与旁边浮动的元素的高度对齐。请运行这个示例：

```html
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <style type="text/css">
.content { max-width: 800px; margin: 0 auto; background-color: #eee; }
.right-panel { float: right; width: 180px; height: 500px; margin: 0 10px; background-color: #5bc0de; }
.main-panel { max-width: 700px; margin-right: 200px; background-color: #5cb85c; }
.bottom-panel { height: 200px; background-color: #f0ad4e; }
.floated-child { float: left; background-color: #d9534f; }
.clearfix:before, .clearfix:after { display: table; content: " "; }
.clearfix:after { clear: both; }
    </style>
</head>
<body>
<div class="content">
    <div class="right-panel"></div>
    <div class="main-panel">
        <h2>Main content title! title title title title title title</h2>
        <div class="clearfix">
            <div class="floated-child">some text here, some text here, some text here</div>
        </div>
    </div>
    <div class="bottom-panel">
        <p>This is bottom panel</p>
    </div>
</div>
</body>
</html>
```

可以看到`clearfix`的那个div的高度与`right-panel`对齐了，调整`right-panel`的高度，`clearfix`的div的高度会跟着调整。

此时我们的修复方案是，将`clearfix`的display属性修改为inline。即
`.clearfix:before, .clearfix:after { display: inline; content: " "; }`


参考：[https://stackoverflow.com/questions/211383/which-method-of-clearfix-is-best]

