---
layout: post
title: 回溯搜索找到游戏解法
date: 2016-07-09 08:06:04 +0800
comments: true
categories: 算法
---

昨晚老婆在家玩游戏，遇到一个关卡，挺有意思，找不到图了，姑且文字描述一下。

* 游戏模型：[1, 1, 1, 0, -1, -1, -1]
* 规则：1或-1可以移动到其旁边0的位置，或者移动到间隔一个障碍的下一个0的位置；1只能向右移动，-1只能向左移动
* 目标：所有-1移动到左边，1移动到右边，即最后状态为[-1, -1, -1, 0, 1, 1, 1]

老婆尝试了好几次无果，给我试试。一看这个游戏，就想到可以用回溯搜索算法来找答案。于是打开电脑开始尝试，得到如下js代码：

```javascript
'use strict'

var _ = require('lodash')
var cards = [1, 1, 1, 0, -1, -1, -1]

var search = function (cards, path) {
  if (_.isEqual(cards, [-1, -1, -1, 0, 1, 1, 1])) {
    replay(path)
    return true
  }
  for (var idx = 0; idx < cards.length; idx++) {
    let moved = move(cards, idx)
    console.log(`moved: ${moved}, path: ${path}`)
    if (moved && search(moved, _.concat(path, [idx]))) {
      return true
    }
  }
  return false
}

var move = function (cards, idx) {
  let moved = _.clone(cards)
  if (cards[idx] === 1) {
    if (cards[idx + 1] === 0) {
      moved[idx] = 0
      moved[idx + 1] = 1
    } else if (cards[idx + 2] === 0) {
      moved[idx] = 0
      moved[idx + 2] = 1
    } else {
      return false
    }
  } else if (cards[idx] === -1) {
    if (cards[idx - 1] === 0) {
      moved[idx] = 0
      moved[idx - 1] = -1
    } else if (cards[idx - 2] === 0) {
      moved[idx] = 0
      moved[idx - 2] = -1
    } else {
      return false
    }
  } else {
    return false
  }
  return moved
}

var replay = function (path) {
  console.log('path found: ', path.join(','))
  var _cards = cards
  console.log(_cards.join(','))
  path.forEach(function (idx) {
    _cards = move(_cards, idx)
    console.log(_cards.join(','))
  });
}

search(cards, [])
```

最终找到的步骤如下：

```
path found:  2,4,5,3,1,0,2,4,6,5,3,1,2,4,3
1,1,1,0,-1,-1,-1
1,1,0,1,-1,-1,-1
1,1,-1,1,0,-1,-1
1,1,-1,1,-1,0,-1
1,1,-1,0,-1,1,-1
1,0,-1,1,-1,1,-1
0,1,-1,1,-1,1,-1
-1,1,0,1,-1,1,-1
-1,1,-1,1,0,1,-1
-1,1,-1,1,-1,1,0
-1,1,-1,1,-1,0,1
-1,1,-1,0,-1,1,1
-1,0,-1,1,-1,1,1
-1,-1,0,1,-1,1,1
-1,-1,-1,1,0,1,1
-1,-1,-1,0,1,1,1
```
