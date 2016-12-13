---
layout: post
title: "在网页显示透明视频"
date: 2016-03-26 21:13:16 +0800
comments: true
categories:
- frontend
tags:
- html5
- game
- opengl
---

最近在一个网站上，看到了很炫的网页特效：视频背景透明。该网址是：[http://videostir.com/](http://videostir.com/)。他们还为用户提供了制作透明视频的服务。用户只需要上传他们要求的格式的视频，就可以生成一个透明的视频。

正如该网站所演示的，这种视频作为网站的引导，效果非常赞，互动的感觉非常强烈。

<!-- more -->

作为一个技术人员，特别是曾经从事过游戏开发的技术人员，很想一探究竟。下面自己实现的一个简单的背景透明效果，仅仅作为技术研究用。

首先，网页视频播放有两个方案：html5的视频播放，flash的视频播放。作为研究用，就不用关注已经过时的flash了。直接看html5下面该如何处理。

html5直接播放视频是没有问题的，可以用来做整个视频背景透明，或者视频缩放，播放控制。很自然的一个想法是，能不能使用html5播放视频，按帧进行截图，然后在canvas上面显示。实际上是可以的。核心代码如下：

```javascript
var width = 496, height = 272;
var canvas = $('#tv-canvas').get(0);
var cxt = canvas.getContext("2d");
var video = $('#tv-video-origin').get(0);

var i = 0, frameCount = 12;
video.addEventListener('loadeddata', function() {
    video.currentTime = i / frameCount;
}, false);
video.addEventListener('seeked', function() {
    i += 1;
    if (i <= video.duration * frameCount) {
        /// this will trigger another seeked event
        setTimeout(function() {
            video.currentTime = i / frameCount;
        }, 1000/24);
        drawCanvas();
    } else {
        console.log('video end');
    }
}, false);

var drawCanvas = function () {
    cxt.drawImage(video, 20, 20, width/2, height/2);
}
```


下面的问题就是canvas上面的图片处理了。这里可以对视频画面做一个要求，对于要透明的视频部分，可以在拍摄视频的时候，找一个纯黑色的背景，这些纯黑色的背景就是需要透明的部分。实际上videostir对视频也是这样要求的。一旦有了黑色透明背景，我们是不是就可以先将提取视频图片，对图片进行预处理，将纯黑色的部分透明化处理，就可以了？添加的代码如下：

```javascript
var cxtt = canvast.getContext("2d");
var video = $('#tv-video-origin').get(0);

var drawCanvas = function () {
    cxt.drawImage(video, 20, 20, width/2, height/2);
    var img = cxt.getImageData(0, 0, width, height);
    var bounce = 40;
    for (var i = 0; i < img.data.length / 4; i++) {
        if( img.data[i * 4] < bounce && img.data[i * 4 + 1] < bounce && img.data[i * 4 + 2] < bounce ) {
            img.data[i * 4 + 3] = 0; // if color is black, set alpha to 0
        }
    }
    cxtt.putImageData(img, 0, 0);
}
```


可以看到canvas上面直接处理视频虽然可以实现效果，但是最终的效果并不是很好，看起来图片质量下降比较严重，而且效率问题也比较严重，在配置低的电脑浏览器上面帧率会更低。

能不能还有其他方法实现这个呢，想到了html5游戏框架。于是在[这里](https://html5gameengine.com/)找了一个：pixi。pixi在检测到浏览器支持webgl的时候，就可以使用webgl来渲染视频，可以看pixi的[demo](https://pixijs.github.io/examples/index.html?s=basics&f=video.js&title=Video)。可以看到游戏框架对视频播放的优化已经做得很好。

pixi是支持webgl的filter的，并且内置了一些常用的filter。但是对于我们这个需求，内置的filter看起来并不够用。于是，我们自己来实现一个filter吧。

关键的shader代码如下：

```
precision highp float;

varying vec2 vTextureCoord;
uniform sampler2D uSampler;

void main(void)
{
   vec2 uvs = vTextureCoord.xy;
   vec4 fg = texture2D(uSampler, vTextureCoord);

   float t = 0.12;
   if (fg.r < t && fg.g < t && fg.b < t) {
        fg.a = 0.0;
   }

   gl_FragColor = fg;
}
```

其原理与js代码里面的类似，直接寻找接近黑色的像素，然后设置其透明度为0。可以明显看到硬件渲染出来的视频已经实现了这个效果，而且视频质量没有下降。

简单分析了一下videostir的实现，他们实现了一个专用的播放器，播放器实现看起来是相当复杂的，会根据浏览器的类型及版本分别采用flash或者opengl来实现。由于没有深入研究他的实现，就不做介绍了。

最后，[这里](/attaches/2016/2016-03-26-transparent-video/index.html)是一个完整的示例，有兴趣的同学们可以研究研究。







