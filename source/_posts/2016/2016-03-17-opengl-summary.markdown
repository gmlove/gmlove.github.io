---
layout: post
title: "opengl知识学习"
date: 2016-03-17 21:04:39 +0800
comments: true
categories: 
---

## opengl es shading language

种类

* 类型：vec2 vec3 vec4 mat2 mat3 mat4 int float
* 结构体：

<!-- more -->

预编译：#ifdef #ifudef #if #endif #define

变量标识：precision midiump lowp in out const attribute(数组常量) varying(从vs传递到fs) uniform(全局变量)


控制语句：

* if ... else ...
* for(..;..;..), while(...), break, return
* discard: 终止执行，只能用在fragment shader中

函数：

* 数组限制 - 作为参数需要指定长度，不能作为返回值


内置函数：

* 访问硬件功能：纹理贴图
* 常用的细小功能：step(返回0,1) clamp mix(x*(1-a)+y*a) smoothstep faceforward
* 硬件加速的某些功能：sin cos