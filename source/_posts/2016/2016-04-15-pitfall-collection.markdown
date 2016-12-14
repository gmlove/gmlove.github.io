---
layout: post
title: "各种坑"
date: 2016-04-15 20:45:41 +0800
comments: true
categories:
- frontend
tags:
- 那些年我们踩过的坑
---

## 重复触发Jenkins build

当使用Jenkins build我们的一个repo的时候，一般我们会想要build master分支。在Jenkins添加git repo的时候，默认添加的监控branch为`*/master`，这个默认的设置就可以满足我们的需求。

但是，事实上`*/master`是可以匹配`master` `xx/master`分支的。如果当前repo里面有一个branch为`xx/master`，那么就会匹配到两个分支。在这样的设置之下，如果master有新的commit，Jenkins就会尝试build这两个分支，于是就会触发两次build。

## grunt在压缩文件的时候，一些自动生成的文件没有包含进去，但当第二次运行编译，文件又被编译进去了

grunt可能在编译之前生成的待压缩的文件列表，由于第一次编译的时候，编译文件没有生成，在压缩的时候就不会包含这个中间文件。第二次编译的时候，中间文件已经存在（可能会在编译过程中更新这个文件），这个时候就可以包含这个文件了。




