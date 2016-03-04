---
layout: post
title: "Maven Tips"
date: 2016-03-04 12:21:46 +0800
comments: true
categories: 
---

## 配置文件

### 继承和聚合

### 属性

内置、POM属性、自定义属性、settings属性、Java系统属性、环境变量

- finalName: 配置最终生成的war包的文件名，可以用于替换默认的${project.artifactId}-${project.version}，便于发布的时候生成合适的路径


## 生命周期

### 三套独立生命周期

- clean: pre-clean -> clean -> post-clean
- default: 
    + validate -> initialize -> 
    + [generate/process-sources -> generate/process-resources] -> 
    + compile -> process-classes -> 
    + [generate/process-test-sources -> generate/process-test-resources] -> 
    + test-compile -> process-test-classes -> 
    + test -> 
    + prepare-package -> package ->
    + pre-integration-test -> integration-test -> post-integration-test ->
    + verify -> 
    + install ->
    + deploy
- site: pre-site -> site -> post-site -> site-deploy

### 绑定生命周期

- 内置绑定
- 自定义绑定


## Plugins

- jetty-maven-plugin 热部署
- maven-release-plugin 版本管理
- maven-gpg-plugin 自动进行GPG签名
- maven-resources-plugin 文件过滤
- maven-site-plugin 生成项目站点
- maven-javadoc-plugin
- maven-checkstyle-plugin
- maven-pdm-plugin 源代码分析工具
- maven-changelog-plugin
- cobertura-maven-plugin 测试覆盖率


## 仓库

- 公共仓库
- Nexus私服


## 版本管理

- 快照版：开发中保持版本稳定
- 稳定版：
    + 所有测试通过
    + 没有快照版本依赖
    + 没有快照版本插件
    + 所有代码进入版本控制系统

### 版本号

1.3.4-beta-2

主版本(架构变更).次版本(较大范围功能变化).增量版本(重大bug修复)-里程碑版本

