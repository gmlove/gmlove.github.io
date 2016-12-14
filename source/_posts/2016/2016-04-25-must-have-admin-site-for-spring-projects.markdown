---
layout: post
title: "Spring项目必备的admin site工具"
date: 2016-04-25 18:05:33 +0800
comments: true
tags: 
- spring
- java
- admin site
---

保持应用状态的可见性对应用的维护和线上问题调试无疑是很重要的；线上进行各种功能开关，日志查询通常也是线上应用运行时必备的功能。这些功能基本上可以放在Admin工具呈现。

对于经典java框架spring，我们是否有一个通用的Admin框架呢，答案是肯定的。

<!-- more -->

实际上spring的Spring Boot Actuator框架为这些常用的功能进行了规范化和实现。目前的功能列表可以在[这里](http://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#production-ready)看到。
很多有用的功能类似环境变量、健康状态、API接口信息、beans状态、各个维度的性能统计数据等等都有覆盖。

使用这些API就可以实现一个功能强大的Admin工具。目前有一个开源的实现[Spring Boot Admin](https://github.com/codecentric/spring-boot-admin)。

但是这个工具也有一些缺陷：

1. 设计的功能有点过多了，甚至包括监控等
2. 需要所有的服务引入新的jar包依赖
3. 需要所有服务配置该服务可访问的地址和admin服务器的地址，如果没有服务自动发现的机制，这个也是比较麻烦的事情

如果我们只需要一个简单的UI界面的话，可以参考我们自己修改过的一个[简化版本](https://github.com/gmlove/spring-boot-admin/tree/only-ui)。这个版本移除了所有的服务器提供的功能直接由客户端来处理所有的数据，服务器提供代理的功能以解决跨域问题。服务只需要引入Spring Boot Actuator。

