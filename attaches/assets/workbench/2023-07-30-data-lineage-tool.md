---
title: 数据血缘分析
categories:
- 数据平台
date: 2023-07-30 20:00:00
---


当你有成百上千行看不懂的ETL代码需要维护时，请看过来。

你面对的ETL代码可能有上千行，可能没有注释，或者注释特别混乱，也可能有很多层嵌套，也可能表和字段可能完全没有业务意义。比如下面这样：

![Messy code](/attaches/assets/workbench/data-lineage/messy-code.png)

你可能非常苦恼要去读懂这样的ETL代码，并在上面进行一些修改。一件令人略宽慰的事情是，很多人正在经历跟你一样的问题。这也是我们团队正面临的问题。

除了去死磕业务背景知识和强行打起精神去读那些天书一样的代码之外，是否还有一些技术手段可以帮到我们？

可能有，那就是数据血缘分析图。

## 数据血缘分析图

下面是数据团队几个技术人员做出来的一个工具，在我们团队中用起来感觉效果不错，能很大程度上提升我们阅读理解天书代码的效率。

下面简要给大家分享一下它的功能：

1.自动解析SQL代码生成可视化血缘分析图

![Parse and Generate](/attaches/assets/workbench/data-lineage/lineage-1.gif)

2.移动到某个字段上面，自动展示该字段计算逻辑

![Display column expression](/attaches/assets/workbench/data-lineage/lineage-2.gif)

3.移动到某个表上面，自动展示该表的构建语句。（截取from及后续语句，并把子查询替换为子查询的名称。）

![Display table expression](/attaches/assets/workbench/data-lineage/lineage-3.gif)

4.可随意拖动表，放大、缩小整个图，并可通过小地图看到全局

![Graph operation](/attaches/assets/workbench/data-lineage/lineage-4.gif)

5.当图非常复杂时，可仅关注某一个选中的字段计算逻辑，而忽略其他部分

![Focus a lineage](/attaches/assets/workbench/data-lineage/lineage-5.gif)

6.当有ETL嵌套时，可选中表，然后跳转到该表对应的ETL血缘分析图

![Jump to dependent ETL](/attaches/assets/workbench/data-lineage/lineage-6.gif)

7.如果有对接元数据系统，可根据ETL中的表字段结合元数据生成ETL文档

![Lineage doc](/attaches/assets/workbench/data-lineage/lineage-doc.png)

8.核心功能还少不了要支持完善的快捷键和中英文

![Shortcut](/attaches/assets/workbench/data-lineage/shortcut.png)

除了这些之外，还有一些高级功能，比如中间临时表移除、多图合并、大量血缘图的管理、基于LLM补充缺失元数据等等，留待有兴趣的同学们探索。

在工具设计上有以下几个原则：

- 尽可能展示100％的ETL代码内容以便做到完全代替源代码的能力
- 保留中间计算过程，以便可以借住代码中的临时表或别名分析数据计算逻辑
- 尽可能捕捉和自动填充注释，以便展示关于某字段的尽可能多的信息
- 在交互上尽可能便于分析人员快速操作，并保持关注

## 试试看

是不是想要试试看？我们将这个工具部署在公司内网的一个服务器上，在这里可以试用：http://10.206.203.10:26600/#data_application/data_lineage/system=sales

如果你在外网，需要连接8082端口的VPN，操作如下：

![VPN](/attaches/assets/workbench/data-lineage/vpn.png)

如果你想测试自己的ETL，可以点这里，输入临时的SQL代码，并生成数据血缘图。

![Test](/attaches/assets/workbench/data-lineage/parse-sql.png)

## 一些实现细节

下面是一些简要的实现思路，供感兴趣的同学了解：

- 借住SqlFluff解析SQL代码为一颗语法树，解析该语法树提取关键信息，并生成一个字段级血缘图
- 使用antv的图组件进行血缘图展示，将上述的图转化为antv的图组建需要的数据结构
- 提供内部的图结构作为API，以便可以进行基于血缘的高级数据分析

## 开发团队

此工具主要由@xiaomeng @guangming @jiajun @yingyi共同开发完成。目前作为数据开发工作台的一部分对外提供功能，并对公司内部开源。如有感兴趣的同学，欢迎联系以上任何一位同学了解细节。
