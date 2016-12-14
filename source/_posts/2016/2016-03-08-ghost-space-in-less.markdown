---
layout: post
title: "Less -- 诡异的空格"
date: 2016-03-08 17:19:49 +0800
comments: true
categories:
- frontend
tags:
- less 
- css
---

Original:

```
.a {
    &.a-b {
        &:hover {
            background-color: #5cb85c;
        }
    }
}

.a-c {
    &:extend(.a .a-b:hover);
}
```

<!-- more -->

Compiled:

```
$ lessc test.less
extend ' .a .a-b:hover' has no matches
.a.a-b:hover {
  background-color: #5cb85c;
}
```

Improved:

```
.a {
    &.a-b {
        &:hover {
            background-color: #5cb85c;
        }
    }
}

.a-c {
    &:extend(.a.a-b:hover);
}
```

Compiled:

```
$ lessc test.less
.a.a-b:hover,
.a-c {
  background-color: #5cb85c;
}
```

css选择器：

* `.a.a-b`表示同一个元素同时包含两个类，
* `.a .a-b`表示子元素选择器，表示当前有`a-b`类并且某一级父元素含有`a`类
* `.a, .a-b`表示多个独立选择器，表示选择当前有`a`类或者含有`a-b`类的的元素

