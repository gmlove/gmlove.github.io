---
title: CTR技术解读 -- 概览
categories:
- 机器学习
tags:
- ctr
- 机器学习
- 深度学习
- 广告点击预测
date: 2020-04-12 20:00:00
---

作为机器学习的一个细分领域，CTR预估一直以来都是研究的一大热点。之所以成为研究的热点，是因为推荐领域巨大的商业价值。无论是我们每天通信用的微信QQ，还是我们每天搜索用的Google百度，或者是娱乐用的抖音斗鱼，广告在这些产品的收入中都占据着非常重要的组成部分，广告收入的背后是广告推荐引擎在发挥作用。除了广告，我们网上购物时的物品推荐，看新闻时的新闻推荐，听音乐时的音乐推荐等等，这些都是推荐引擎发挥作用的地方。在这些地方，推荐引擎都产生了巨大的价值。

对于如此重要的一个领域，我们做机器学习的小伙伴多多少少都需要了解一下。下面我将结合一个项目上的实际案例，来分享一下整个CTR的研究和应用情况。我将尝试尽可能用通俗易懂的语言，使得只要有一些基本的机器学习知识就能理解文章内容。文章将从应用角度出发，重点关注基本原理及工程实现而弱化更偏理论的公式推导等。

本系列文章将分为以下几个部分：

- 概览：背景，研究的问题，技术架构及演进，一个小的例子
- 传统模型：协同过滤、LR模型及其应用
- Embedding、深度学习模型及其应用
- FM与DeepFM
- DCN及其他深度模型
- 基于LSTM的模型：DIN、DIEN及DSIN
- 多任务模型

<!-- more -->

## CTR

CTR的全称是Click-Through Rate，即点击率。这里的点击当然就是指点击推荐的内容，如视频、音乐、新闻、广告等。

如何进行推荐呢？从结果来看，我们希望算法能输出一个推荐内容的列表，这个列表按照一定的规则（比如用户感兴趣的程度、最大化盈利）进行排序。为了计算这个列表，用户点击某个内容的概率将是一个重要的输入。所以，CTR预估成为了推荐领域的最重要的部分之一。事实上，CTR预估的准确程度很大程度上影响着推荐结果的好坏。

## 主流推荐系统架构

在推荐场景中，我们需要思考和解决下面这些问题：

1. 如何处理数据形成特征
2. 如何高效的、稳定的进行模型训练
3. 如何将训练好的模型快速用于线上推理
4. 如何支持模型在线优化
5. 如何快速的实验新的特征及模型

从整体上来看，推荐系统架构将是一个离线、近在线、在线处理相结合的一个复杂系统。完全在线处理推荐系统中的所有问题是一种理想情况，其成本将非常高昂，即便大型互联网公司也未必能做到。一个相对完善的推荐系统解决方案都会组合利用这三种模式各自的优势来解决问题。

一般的机器学习任务都会包含数据和模型两部分，推荐系统也不例外。从数据层面看，推荐系统的数据可以来自三个方面：1. 用户数据，比如年龄、性别等；2. 物品（或广告）数据，比如广告文案、图像、视频等；3. 场景数据，比如广告位、当前时间、是否节假日等。于是，我们可以得到这样的推荐系统逻辑架构：

![推荐系统逻辑架构](/attaches/2020/2020-04-12-an-apprehensible-way-to-describe-ctr--introduction/logic-architecture.png)

对于这些数据，如果按照实时性维度来看，可以分为离线数据、近实时数据、实时数据。正好就可以用不同实时性的数据处理工具进行处理。

从模型层面看，简单的处理方式就是离线模型训练配合在线模型定期更新。如果模型训练时间比较长，可能只能实现按天进行更新。如果模型可以有一定的迁移能力，那么就有可能按小时进行模型离线迁移学习，实现按小时进行模型更新。如果工程能力比较强，能同时实现模型在线训练和服务，那么模型的更新周期就能缩短到分钟级别。模型更新的时效性通常对推荐的效果有很大的影响。

一般而言，当前主流的推荐系统会在设计上将整个推荐过程分成这样几个主要的步骤：召回 -> 排序 -> 再排序。召回层主要目标是快速缩小推荐范围，一般会利用业务规则、高效的算法或简单的模型实现。而排序层则对筛选出来的小规模数据进行CTR预估和排序。再排序层则会充分考虑推荐结果的多样性、流行度、新鲜度等指标，对排序好的结果进行最终的调整。

有了以上这些分析，我们可以得到一张常见的推荐系统的架构图：

![推荐系统架构](/attaches/2020/2020-04-12-an-apprehensible-way-to-describe-ctr--introduction/recommendation-system-architecture.png)

## 推荐系统技术演进

在推荐系统中，模型发挥作用最明显的一步就是排序层，同时排序层也是整个推荐系统中最重要的一步。所以，大多数的研究都是针对排序层的研究，这里提到的技术演进也主要是指排序层模型的技术演进。

2010年之前，业界主流的推荐系统一般都是基于协同过滤或简单的逻辑回归（`LR`）模型实现，也有的系统会基于某些业务特征进行推荐，比如标签、地域、热度等。这些模型利用数据的能力有限，需要大量的领域专家设计特征，准确率也不高。

从2010到2015年间，随着移动互联网的高速发展，基于传统模型进行了大量的改进。出现了因子分解机（`FM`）、梯度提升树（`GBDT`，`Facebook`的主要模型）等基于协同过滤的改进模型。也出现了`FTRL`（`Google`的主要模型）、混合逻辑回归（`MLR`，阿里的主要模型）等基于LR模型的改进。

2015年之后，随着深度学习在计算机视觉领域和自然语言处理领域的成功应用，推荐领域的研究方向也快速转向了以深度学习为主。由于深度模型更少的依赖人工特征工程，并能更有效的利用数据和发现数据中的模式，这些模型得到了非常快速的迭代演进和工程应用。当前常见用于各大互联网公司的模型，比如`Google`提出的`Wide&Deep`模型，微软提出的`Deep Crossing`模型，阿里提出的引入注意力的`DIN`模型及可以进行时序建模的`DIEN`模型，基于深度学习的协同过滤模型`NeuralCF`等等。可以看到，深度推荐模型的设计思路开阔，方向多样，呈现出百花齐放的状态。

简单总结一下可以得到下图：

![推荐模型发展](/attaches/2020/2020-04-12-an-apprehensible-way-to-describe-ctr--introduction/model-evolving.png)

后续文章中，我将会陆续挑选一些主流的模型进行分析和实践。

## 一些重要的问题

除了模型之外，推荐系统中还有一些比较重要的问题。这里简要分析一下。

### 特征处理

机器学习领域有一个基本的认知，那就是“数据和特征决定了机器学习的上限，模型和算法只是在逼近这个上限”。所以，好的特征是非常关键的一步。在优化模型的时候，如果陷入了一个瓶颈，回过头来从数据层面进行分析可能会有意想不到的效果。常常需要对数据和模型进行综合优化才能实现较大的性能提升。

为了将特征接入模型，需要对数据进行一定的处理，转化为模型需要的格式。对于一般的类别特征，常见的特征处理办法是进行`OneHot`编码。对于多类别的特征，比如近期观看的电影列表这类，进行`MultiHot`编码可以实现类似`OneHot`的效果。（这两类数据处理方式可以参考我的[另一篇文章](http://brightliao.me/2020/01/21/spark-performance-tuning-on-billions-of-data/)。）对于一些统计特征，比如平均每月在产品中的花费等，则可以直接将数值特征输入模型。

除了这些基本的数值型或类别型数据，由于深度学习的快速进步，对于文本、图片、视频等数据现在也有了更多的处理办法。比如对于文本特征，可以用数据挖掘常用的`TF-IDF`进行特征处理，也可以用深度学习模型，如`Word2Vec`、`Transformer`等模型将数据编码为稠密的特征向量。对于图片和视频，同样也可以采用深度学习模型提取特征向量。这些数值化之后的特征就可以作为模型输入了。除了应用基本的深度学习模型，在推荐领域，还发展出了`Item2Vec` `Graph Embedding`等抽取特征的模型。虽然这些特征的处理过程代价较高，但是它给我们带来了大量新的应用数据的可能。

### 效果评估

推荐问题，其本质是一个二分类机器学习问题，因为只需要预测用户会不会点击某个物品即可。为了支持排序，模型需要输出一个物品被点击的概率。对于这样的问题，一般的评估指标是模型的准确率（Accuracy）。但是在推荐领域，我们不能使用简单的准确率来评估模型效果。

推荐领域的数据集一般是非常不均衡的。对于应用给我们推荐的物品，大概计算一下点击过的次数就能意识到这个问题。有可能推荐的物品中有99%是没有被点击过的。这就带来了训练数据集庞大而非常不均衡的情况。在这样的情况下，如果采用准确率来评估，只要模型对所有数据预测为负就可能得到99%的准确率。但是此时的模型几乎没有任何用处。

在推荐领域，常用的模型评估指标是`AUC`，即 Area Under Curve 。是什么曲线下面的面积呢？在推荐领域这里的曲线一般是指`ROC`曲线，`ROC`曲线是以`FPR`（False Positive Rate，假阳率，错误的判定为正例的概率）为横轴，以`TPR`（True Positive Rate，真阳率，正确的判定为正例的概率）为纵轴的一条曲线。由于在`AUC`评估指标中使用了百分比为基础数据，它的结果与样本是否均衡无关。

除了模型指标之外，还可以考虑的一个问题是如何尽可能利用实时的数据进行效果评估。一个常用的方式是进行`A/B`测试，通过将用户进行随机分组并对不同的分组应用不同的模型，然后评估不同模型的效果。除此之外，微软在2013年提出了`InterLeaving`的方式进行在线的效果评估。`InterLeaving`通过在同一推荐位上交替展示不同模型的推荐来对比不同模型的效果。有了这些实时的模型评估手段，我们可以更有信心的进行模型选择，甚至可以让模型选择自动化的进行。

### 冷启动

推荐领域的一个重要的问题是冷启动问题。当一个新用户进入的时候，没有历史数据，如何进行推荐呢？当新物品加入的时候，如何进行推荐呢？当系统刚开始搭建起来，应用刚进入市场，如何进行推荐呢？

通常可以有两类方法解决冷启动问题。一是充分利用业务进行推荐，比如新用户加入的时候可以推荐热门榜单、最新榜单等条目。二是尽可能的挖掘信息，充分利用已有数据进行推荐。比如当新用户进入的时候，我们往往可以获取到用户注册时的基本信息，如人口学特征等，然后我们可以根据这些信息查找相似的用户，利用这些相似用户的信息进行推荐。当新物品加入的时候，也可以采用类似的方式。引导用户输入更多信息是一种有效的挖掘信息的方式，比如在产品设计上可以在新用户加入时引导其选择兴趣爱好，在音乐、新闻等内容应用上经常能见到此类功能设计。从第三方平台获取数据也不失为一种手段，当前有不少的统计分析平台，可以获取用户的一些统计特征，这些数据也可以作为推荐系统冷启动的数据来源。

### 探索与利用

一些做的不够完善的推荐系统常常在一段时间之后陷入推荐内容过于同质化的问题。比如系统发现用户喜欢汽车，就频繁推送汽车相关的物品。同质化严重将导致用户对推荐的内容逐渐失去兴趣。

这里的问题其实就是多度利用了历史数据带来的问题。推荐系统除了做好数据利用，还需要考虑如何探索用户的新需求新兴趣，同时也需要考虑长尾物品或长尾物品的推荐。在推荐系统的再排序过程可以根据业务具体情况加入一些合理的条目。比如经典的`ξ-greedy`探索策略，以一定的递减的概率随机加入一些推荐条目。也可以在模型层面来考虑解决这个问题，比如可以有针对性的添加模型参数随机扰动机制，引入了强化学习思想的`DRN`模型可以作为此类方案的一个典型代表。

## 一个简单的示例

实现一个完善的推荐系统是一个复杂而巨大的工程，但是，在理论和工具都非常丰富的现在，要搭建一个可用的推荐系统`MVP`其实并不难。下文将以一个简单的示例作为结尾。

协同过滤是常见的推荐系统算法，其基本思路是根据相似性进行推荐。例如，基于用户的协同过滤算法会推荐跟目标用户有相似爱好的其他用户所喜欢的物品。下面我们将在[`MovieLens`](https://grouplens.org/datasets/movielens/)数据集上面进行实验，尝试实现基于用户的协同过滤推荐系统。

```python
import numpy as np
import pandas as pd


def normalize(vec_a: np.ndarray, vec_b: np.ndarray):
    vec_a_avg = vec_a.sum() / (vec_a > 0).sum()
    vec_b_avg = vec_b.sum() / (vec_b > 0).sum()
    vec_a_intersect = vec_a * (vec_b > 0)
    vec_a_intersect[vec_a_intersect > 0] = vec_a_intersect[vec_a_intersect > 0] - vec_a_avg
    vec_b_intersect = vec_b * (vec_a > 0)
    vec_b_intersect[vec_b_intersect > 0] = vec_b_intersect[vec_b_intersect > 0] - vec_b_avg
    return vec_a_intersect, vec_b_intersect


def cos_sim(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


if __name__ == '__main__':
    ratings = pd.read_csv('data/movielens/ml-latest-small/ratings.csv')

    movies = dict([(movie_id, idx) for (idx, movie_id) in enumerate(ratings['movieId'].unique())])
    print('movies({}): {}...'.format(len(movies), list(movies.items())[:10]))

    user_rating_vectors = {}
    for user_id in ratings['userId'].unique():
        user_rating_vector = np.zeros(len(movies))
        user_ratings = ratings[ratings['userId'] == user_id]
        for (_, rating) in user_ratings.iterrows():
            user_rating_vector[movies[rating['movieId']]] = rating['rating']
        user_rating_vectors[user_id] = user_rating_vector

    print('user_rating_vectors({}): {}...'.format(len(user_rating_vectors), list(user_rating_vectors.items())[:2]))

    target_user_id = 610

    target_user_rating_vector = user_rating_vectors[target_user_id]
    user_sim = [(user_id, cos_sim(*normalize(target_user_rating_vector, rating_vector)))
                for (user_id, rating_vector) in user_rating_vectors.items()
                if user_id != target_user_id]
    user_sim = sorted(user_sim, key=lambda x: x[1], reverse=True)

    similar_user = user_sim[0][0]

    print('similar_user: {}, user_sim: {}...'.format(similar_user, user_sim[:10]))
    print('target_user_rating_vector: {}\nsimilar_user_rating_vector: {}'.format(list(target_user_rating_vector), list(user_rating_vectors[similar_user])))

    similar_user_ratings = ratings[ratings['userId'] == similar_user]
    similar_user_ratings = similar_user_ratings.sort_values(by='rating', ascending=False)

    print('recommended movies:')
    print(similar_user_ratings.head(10))
```

到这里大家应该了解了一些基本的推荐系统知识，后续文章将挑选几个模型进行介绍并实践。敬请期待！











