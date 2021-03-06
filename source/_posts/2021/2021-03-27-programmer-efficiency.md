---
title: 成为高效的程序员
categories:
- 心态
- 敏捷
tags:
- 效率
- 工具
- 编程思想
date: 2021-03-27 20:00:00
---

## 一个故事

最近有一次我观察项目组中一位经验较浅的小伙伴写代码，发现：

1. 快捷键用得少，缺乏一点去研究快捷键的意识
2. 自动补全功能没有充分利用，基本靠记忆+手敲
3. 使用mac上并不好用的键盘

这带来的结果就是手上的速度跟不上思维的速度。

<!-- more -->

## 高效开发的起点：打字速度和快捷键

常常记起，与我的buddy在一个项目组时，他常常会提醒我们如何高效的做开发。比如：手速。他推荐并常常组织我们在一个练习打字速度的应用上面练习和比赛。

快捷键的使用更是ThoughtWorks开发人员的传统强项，与有经验的TWer pair开发就能体会到如何完全不用鼠标而把键盘敲到飞起带来的超高编程效率。TDD、重构、格式调整等等我们日常的开发动作在高效的快捷键使用下更是相得益彰。

我曾经在一个客户的咨询项目上，有一个客户的开发同学，见面就聊到说他也来参加过ThoughtWorks的面试，但是最后没有进来。这位同学提到这件事显得很不服气，说他了解到没有进来的原因之一居然是快捷键用的不熟，而且很自以为是的补充说：快捷键跟开发能力有什么关系？介于他是客户方的人，我没有当面的跟他争论，但是其实当时我就在心底里暗道：还好没进TW！

快捷键的确不是衡量一个好的开发人员的唯一标准，但是快捷键的灵活使用是一个高效开发人员的重要标志。作为一个开发人员，虽然不必刻意追求打字速度和快捷键使用，但是至少应当心向往之才是。

与此相关的，我不禁又想到咱们的老TWer熊节的课程《学会TDD，十倍提升编程效率》。只有TDD可能并不能十倍提升效率，但是TDD结合超快的打字速度和熟练的快捷键使用，就很有可能了。

## 高效开发的故事

高效的开发者是可以快速完成很多事情的。我自认为不算一个特别高效的开发人员，但是我可以列举一下曾经做过的一些个人认为还算高效的事情，可以供大家参考。

### Happy Bird 的实现

在14年的时候，我想转型做游戏开发。当时的背景是以后端开发为主，也偶尔兼职做一些前端的小功能。于是我找了一个周末，研究了一下iOS的开发。正巧当时有一个小游戏`Flappy Bird`，挺火的，但是游戏设定又比较难。好吧，那就试试看重新实现一下这个游戏，调整一下难度吧。于是，我花了不到两天的时间做了一个`Happy Bird`出来，设置难度之后，可以轻松的玩很长时间。

怎么做的呢？大致有这三个步骤：

1. Root一下我手上的iPad，下载安装好`Flappy Bird`，找到其安装程序路径，把它的资源文件拿出来。
2. 研究iOS上面的动画开发，看了几个小例子
3. 编写代码实现`Happy Bird`

核心的代码在这里，有兴趣的小伙伴可以看看：https://github.com/gmlove/happybird/blob/master/Classes/HelloWorldScene.cpp

### 项目中的一些实践

最近我觉得也有几个可能值得一提的例子。

我们项目是为客户构建一个企业内部的数据平台，平台基于Horton Works（已与Cloudera合并）的开源的Hadoop发行版HDP构建。

由于HDP毕竟是一个完全开源的产品，直接在客户环境上使用，总还是显得有点不够。缺什么呢？一是一个入口的平台Portal页面，通过这个页面可以导航到具备各项功能的应用中。比如，元数据管理要导航到`Atlas`，权限管理要导航到`Ranger`，平台运维要导航到`Ambari`。

刚开始的时候，项目把优先级安排在了进行数据接入、指标开发等等更具业务价值的地方。于是这个入口页面就一直被搁置了。项目中的dev们由于要经常使用这些工具，大家都需要通过加浏览器书签的方式进行管理。BA PM等角色由于不需要关注这些工具，也没有一个入口可以进入这些工具进行操作，于是项目前期阶段几乎不了解这些工具是干什么和怎么用的。这样一来，由于缺乏这样一个入口界面，数据平台总是没有数据平台的感觉，给人的印象是一堆散落在各个地方的工具，没有体系化。同时，这造成了整个团队信息共享不畅，效率降低。

我观察了这个现象，感觉得做点什么。

我开始评估了一下。我们如果把这个页面对应的故事优先级提高，对项目整体安排是不利的，这是其一。其二是，就算把这个优先级调高了，按照我们的敏捷项目管理和实践，要完成这个，需要经历BA的业务分析，UX的界面设计，DEV的估点，最后才能到开发。整个流程下来周期比较长，为了这么简单一个页面，似乎并不值得。

于是，我找了一张故事卡，感觉这个故事卡做完，和预估的开发时长相比会有一些空间。好吧，说干就干，我利用了这一点空间，使用还算比较熟悉的Ant前端组件库，基于React框架，参考Ant的应用样例设计，做了一个静态的页面。这个页面做完，总用时差不多是3个小时，包括代码库初始化，找图，样式微调，把各个工具的通过菜单组织到一起，编写运维脚本，通过nginx进行静态部署等操作。

这个portal页面大概长成下面这样:

![platform portal](/attaches/2021/2021-03-27-programmer-efficiency/portal.png)

完成之后，我把这个页面发到项目组的群里面，让大家体验。最终的效果可能未必能给大家带来多大的效率提升，但是这个MVP的页面在我看来对项目的推进起到了很好的作用。一是辅助进行了ShowCase，让客户看到了我们整个数据平台是一个体系化的平台。二是成为了Portal页面设计的参考原型，为BA设计这个页面提供了参考。三是方便了大家知道和查找相应的工具。

上面的例子中，产出并不多且不显眼，而内容还稍微有点长，大家读下来可能会觉得有点累。但是项目中的事情总是复杂的，有很多的背景，不仅要考虑客户的感受，还要考虑如何有智慧的与团队其他人员合作，可能未必能有自己独立自由的做一些开发工作来得简单和自在。其中的原因相信大家也可以理解。

除了上面的例子，还有很多其他例子，比如:

- 我们实现了一个自动生成etl代码的工具
- 我们用hexo把数据标准的excel文档转换为一个静态网站
- 我们用mkdocs来快速制作数据平台的使用文档
- 我们利用squid反向代理解决了客户需要访问很多台节点带来的复杂网络配置问题
- 我们配置了es搜集了squid日志以便可以进行完整的用户行为分析
- 我们将etl的统计数据发送到es以便我们可以更简单的做出一些必要的etl任务报表给大家诊断问题提高效率
- 我们用docker来隔离开发人员的Hadoop集群使用环境
- ...

## 敏捷与高效开发

上面这些东西很多看起来并不是直接的具备业务价值的，在做这些事情之前，其实也很难去规划出这么多的故事卡，所以可能难以通过敏捷的项目管理来实现。

那它们是怎么实现的呢？我认为是另一种敏捷的形式，并不是Scrum敏捷项目管理，而是从开发人员的视角出发的一种敏捷。由高效的开发人员，通过对于效率的不懈追求，充分利用各种可以利用的技术，不断重构，既包括代码的也包括工作方式的重构，自发的构建出的一套系统。

这并不奇怪，如果我们去翻看敏捷的起源，就会发现敏捷其实是由一些优秀的开发人员，通过思考总结提炼，把高效的习惯沉淀下来而成的。我们从敏捷前辈大师们的大作中就能看出这些，如Martin Fowler的《重构》，Neal Ford的《卓有成效的程序员》、《函数式编程思维》，Eric Evans的《DDD》，Kent Beck的《XP极限编程》等等。大师们对于细节的不苟，对于高效的执著，对于软件艺术的追求，是这些大力推动了软件开发的变革。

同时，如果我们去看github或者apache下的诸多开源软件，我相信这些软件的雏形或者MVP多是由某些开发人员为提高效率而自发开发的。特别是在前期，这些软件是很难由某一个较为完整的团队组织起来开发而成。

当然这里并不是否定Scrum的价值，在一个敏捷团队中，Scrum的团队管理思路当然也是很有效的。

## 最后

讲了这么多，作为一个十年从业经验的开发人员，同时作为一个五年的TWer，其实是想给广大开发人员分享一下这些经验。希望我们可以一直保持对高效的软件开发的追求，对不断提升自己的追求，对软件艺术的追求，对用软件技术为客户创造价值的追求。

最后，还有一个有意思的故事给大家分享。有一次，我们的BA同学给大家做读书分享，提到了一本不赞成敏捷实践的书，里面讲到，他们的工作方式是：1. 提出一个大的愿景 2. 让2-3个开发人员组成一个小组基于这个愿景来做开发 3. 利用1-2个月的时间产出一个可用的产品。在他们看来，这一方式是比当前的我们做的敏捷更为高效的软件开发方式。不知道大家是怎么看的，我印象很深的是另一个经验丰富的项目组成员的话：对于一些经验特别丰富的人，特别厉害的人，任何规则都是无效的，只能降低效率，而提供一个环境让他们独立的以自己的方式做事，就足够让他们做出一些令人惊奇的软件了！


## 材料

对于想练习自己的打字速度的同学，可以参考这个工具：https://dazi.kukuw.com/ 我在上面第一次打字速度如下，相信大家可以轻易超越，特别是年轻的小伙子们，大家手速一定不一般。

![typing speed](/attaches/2021/2021-03-27-programmer-efficiency/typing.png)

















