---
layout: post
title: "微服务架构总结"
date: 2016-04-26 18:53:02 +0800
comments: true
categories:
- micro service
tags:
- micro service
---


## 概述

From Martin Fowler [microservices](http://martinfowler.com/articles/microservices.html):

> 微服务架构即是采用一组小服务来构建应用的方法。
> 每个服务运行在独立的进程中，不同服务通过一些轻量级交互机制来通信， 例如 RPC、HTTP 等。
> 服务围绕业务能力来构建，并依赖自动部署机制来独立部署。

From Sam Newman [Building Microservices]:

> You should instead think of Microservices as a specific approach for SOA in  the same way that XP or Scrum are specific approaches for Agile software development.

微服务即SOA的一种实现方式。企业服务总线（ESB）设计的失败给SOA带上了负面的标签。

<!-- more -->

## 特征

* 组件服务化
* 按业务能力组织服务
* 服务即产品: You built it, you run it
* 技术栈和数据去中心化
* 基础设施自动化
* 容错设计
* 兼容设计

## 实施

* 前提：复杂度低于零界点，可能导致部署工作量上升
* 拆分：业务能力
* 协作：契约文档
* 测试：[测试四象限](/attaches/2016/2016-04-26-about-micro-service-architecture/test-dimension.png) [测试金字塔](/attaches/2016/2016-04-26-about-micro-service-architecture/test-triangle.png)
* 部署：虚拟化或容器等隔离技术，每一个service一个主机
* 监控：基础监控（网络，磁盘，os） 服务监控（响应时间，TPS） 业务监控（多维度，长时间）
* 原则：[战略目标 架构原则 设计与交付实践](/attaches/2016/2016-04-26-about-micro-service-architecture/principles.png)

## 角色变化

* 普通工程师：仅仅开发功能 -> 开发、运营服务
* 每个服务至少有一个工程师作为负责人，能力更强的人可能会负责更多的服务
* 开发人员交集减少，大规模的团队并行开发好处明显
* 对个人能力要求更高，个人成长路线的发展也打开了空间

## 参考

https://segmentfault.com/a/1190000004998167
