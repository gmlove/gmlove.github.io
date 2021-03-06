---
title: 上山容易下山难
categories:
- TDD
tags:
- 敏捷
- TDD
date: 2019-12-22 20:00:00
---


昨天和项目组的几个小伙伴去爬山。这次爬山坐标深圳梧桐山。从西北门进山，我们沿着蜿蜒的公路一路而上，历时三小时登顶。下山时不想原路返回，故意选了另一条路。从山上往下看，这条路陡峭得很。不过，对于几个血气方刚的男性，这条路正好。因为大家都觉得上山太简单不过瘾，想挑战一下高难度。于是我们一致同意换路而行。我们走的这条路就是凌云道。

说起凌云道这条路，真是名副其实。它不仅几乎呈垂直高度下降，而且台阶较窄仅有半脚宽。路两边树丛虽然茂盛，但是都不高大，山下的情形总是能进入眼帘。我一向有点恐高，在这垂直高度近1千米地方往下看，还真是心里有点慌。刚开始下山，小伙伴们见此山势，纷纷停下脚步拍照。得几张无P图片如下，大家可以感受一下：

<!-- more -->

![lingyundao](/attaches/2019/2019-12-22-its-harder-to-go-downhill/lingyundao-1.jpeg)
![lingyundao](/attaches/2019/2019-12-22-its-harder-to-go-downhill/lingyundao-2.jpeg)
![lingyundao](/attaches/2019/2019-12-22-its-harder-to-go-downhill/lingyundao-3.jpeg)

其实二维图片由于没有了层次感，跟实际感受相比还要差很多。如有爱爬山的小伙伴，建议去试试亲身感受一下（千万注意安全）。

大家开始下山，其他几个小伙伴都按照我们正常向前走的方式下山。我由于之前爬山偶然得到一个经验，那就是倒着下山，后续几天小腿不会那么痛。于是我就倒着走下山。一路下山的还有很多其他路人，大家几乎都是按照正常的向前走的方式下山。这个时候估计有不少人笑话我这种下山方式很奇怪。不过，我是基本上不在乎大家怎么看的。因为我相信不管我们自己的做法多怪异，只要不跟他人的切身利益相关，在别人眼中其实都是无关紧要的，最多一笑而过。

不过，一路下山，我还是很愿意和小伙伴们分享一下倒着下山的感受的，因为我真的感受到这样走下山是又省力又安全。特别是在下山了几百米之后，好几个小伙伴都在说小腿在颤抖的时候。跟大家分享之后，引来了几个小伙伴的尝试，大家确实也发现有一定的效果。

我仔细思考了一下为什么我会感觉倒着下山能既省力又安全，可能是这样几个原因：

- 面向山体，而不是面向山下，可以避免恐高带来的心慌
- 即便脚下踩滑，手上可以快速使劲，不至于滑太远甚至摔下去
- 每下一步都是先由上面的一只脚使劲稳住，然后再由另一只脚伸直下探，最后下探的脚尖着地支撑。这样的动作组合不会给下山的脚带来太大的冲击力，进而避免运动过度而导致后续小腿痛（我查了一些资料，发现确实是有科学根据的，从膝盖受力来说，下台阶几乎是上台阶时的两倍）

今天已是爬山回来的第二天。原以为昨天爬山一整天，即便下山倒走也不可避免的会小腿痛，但今早起来居然惊喜的发现仅仅是双腿有点累需要恢复一下而已，小腿完全没有疼痛难以走路的感觉。尝试一跳，还能跃起。

惊喜之余，我又自问了一下，为什么刚开始我可能也不会用这种方式。大概有这样几个原因：

- 倒走姿势怪异，在别人眼中成为了另类
- 挪步缓慢，速度跟不上
- 形似年迈老人的姿势，无法展现年轻人的活力
- 挪步的时候，需要向后看身后的台阶，但刚好被身体挡住，很多时候需要靠估计来挪步，不够安全
- 可能由于强度减弱，达不到爬山健身的目的

想到这里，我不禁发现倒走下山似乎跟我们的TDD实践有一定相似之处。刚接触TDD的小伙伴们，估计大部分也会这么觉得：

- TDD姿势怪异，是另类
- TDD缓慢，速度跟不上
- TDD过于强调细节，每写一行代码都要先写一行测试，过于谨慎，过于冗余
- TDD可能导致测试过多，难以维护
- TDD关注在测试上，可能导致设计关注不够，从而产生设计很差的代码

初看TDD，确实会有违反常人先写代码后写测试的逻辑，我们会觉得不舒服，效率降低。有人甚至会觉得这完全是倒行逆施，有很大的抵触心理。这大概也是为什么TDD一直以来存在这么大的争议。

但是，TDD却是作为敏捷最核心的实践之一存在着。TDD既然这么不舒服，它究竟是为什么能得到这一部分人的认可，并得到他们不遗余力的推广呢？也许只有当我们了解他们的感受，理解这些人背后的逻辑之后，我们再来审视这件事，可能才能消除自己先前的偏见。就像倒走下山，一开始我们可能不会认可，自己随意尝试一下，我们可能还是不会认可。也许只有我们真的深入尝试过体会过（比如某一次坚持倒走下山，然后第二天体会一下），然后我们再来看待这件事才能有自己的答案。浅尝辄止就给出判断是不够的。

事实上，一个让人用起来不舒服的实践是很难一直得到人的认可的。即便当时强行接受，也几乎无法让人每天坚持去做。只有当一个人深刻认识到这个实践的好处，觉得它用起来很舒服，在潜意识里面认可它，我们才可能每天坚持去做。想想我们每天早上起来洗脸刷牙这件事，小时候吵着闹着不舒服不想洗，但长大后居然就变得自觉起来，而且一直坚持在做。我们之所以能坚持，是不是主要由于我们觉得洗完脸刷完牙之后感觉特别舒服，让我们有了新的面貌去迎接新的一天？

所以我们要想做到每天坚持TDD实践，只有当我们潜意识都认为TDD实践很舒服的时候，我们才能做到。因为这时它已经变成一种自然发生的事情，跟意志力无关。要让我们潜意识里认识到这一点，需要每个人自己去深入体会和改进这个过程中的每一步。就像我在体会倒走下山的每一步的时候一样，当我觉得倒走下山的每一步都很舒服的时候，我自然会每次都倒走下山了。开始时我也会觉得看不到身后的台阶而不舒服，但是我可以不断的调整姿势来改进这一点，直到我觉得舒服为止。对于TDD，我们可以尝试自问一下：TDD每一步都让我们很舒服吗？我在哪一步会不舒服？为什么会不舒服？是不是我的理解有问题？是不是我的方式不对？能不能改进这个方式，让我用起来舒服？

我相信一旦大家这样自问并且自我改进之后，一定能找到一种自己用起来很舒服的TDD的方式并坚持下去。

在这里我也趁此机会分享一些我个人的经验。

明确目标。TDD不是做软件的目标。就像倒走下山一样，怎么走不重要，关键在于我们能不能下山，能不能高质量的下山。同样，做出软件，做出高质量的软件是我们的目标。那么，我们要关注的事情变成：TDD是如何去支持这一目标的。从这一点来说，如果我们发现我们做的TDD中的某一步骤不能支持这一目标，那就是可以改变的。

以设计为核心。软件的内建质量几乎完全通过设计来呈现。软件的设计从大到小可以包括架构设计、模块设计、类设计、方法设计、变量设计。TDD如何帮助我们改进这些不同层面的设计呢？它在这些设计的过程中发挥着什么样的作用？如果能回答这两个问题，那么我们实践TDD就不是问题了。

关于以上两个方面我也有一些自己的思考，如果大家有兴趣，请参考我之前的博客：

- [从改善设计的角度理解TDD（一）](http://brightliao.me/2019/07/20/tdd-for-improving-design/)
- [从改善设计的角度理解TDD（二）](http://brightliao.me/2019/08/18/tdd-for-improving-design-2/)

上山不难下山难。上山几乎成为了大家日常可以完成得很好的事情，无需过多的技巧。但是对于下山，我们可能真正需要认真思考一下方法。





